"""
Extract Llama-3.2-1B last-token embeddings for every chunk × layer × N.

Output: cache/embeddings.npz with arrays
    X[mode][N]      -> (num_chunks, num_layers, hidden_dim) float32
    y[mode]         -> (num_chunks,) string labels
    sources[mode]   -> (num_chunks,) strings
    layers          -> list[int]   (index of each layer kept)
    Ns              -> list[int]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

MODEL_NAME = os.getenv("MODE_PROBE_MODEL", "meta-llama/Llama-3.2-1B")
# Llama-3.2-1B has 16 transformer blocks -> 17 hidden_states with embedding layer.
# We extract all layers.

# Token-context lengths to evaluate (Sarfati varies N ∈ {8,16,...,128}).
DEFAULT_NS = [16, 64, 128]


class ChunkDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, N: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.N = N

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        # Tokenize and take first N tokens (no special tokens).
        ids = self.tokenizer.encode(text, add_special_tokens=False, max_length=self.N,
                                    truncation=True)
        # If sample short, drop it later. Pad with EOS to N for batching, but track real_len.
        real_len = len(ids)
        if real_len < self.N:
            ids = ids + [self.tokenizer.eos_token_id] * (self.N - real_len)
        return torch.tensor(ids, dtype=torch.long), real_len


def collate(batch):
    ids = torch.stack([b[0] for b in batch], dim=0)
    real_lens = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return ids, real_lens


@torch.no_grad()
def extract_for_mode(df_mode: pd.DataFrame, model, tokenizer, device, Ns,
                     batch_size: int = 16, fallback_model_name: str | None = None):
    """Return dict[N] -> (num_chunks, num_layers, hidden_dim)."""
    out = {}
    for N in Ns:
        ds = ChunkDataset(df_mode, tokenizer, N=N)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
        all_embeds = []
        for ids, real_lens in loader:
            ids = ids.to(device)
            outputs = model(ids, output_hidden_states=True, use_cache=False)
            # hidden_states: tuple of (num_layers+1) tensors (batch, seq, hidden)
            hs = torch.stack(outputs.hidden_states, dim=1)  # (B, L+1, T, D)
            # Select last *real* token per sample (real_lens-1).
            B = hs.shape[0]
            # Clamp to [0, N-1].
            idx = (real_lens - 1).clamp(min=0).to(device)  # (B,)
            # gather along sequence dim
            idx_exp = idx[:, None, None, None].expand(-1, hs.size(1), 1, hs.size(3))
            last_token = hs.gather(2, idx_exp).squeeze(2)  # (B, L+1, D)
            all_embeds.append(last_token.cpu().float().numpy())
        out[N] = np.concatenate(all_embeds, axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--Ns", type=int, nargs="+", default=DEFAULT_NS)
    parser.add_argument("--input", default=str(DATA / "chunks.parquet"))
    parser.add_argument("--output", default=str(CACHE / "embeddings.npz"))
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} chunks across modes: {df['mode'].unique().tolist()}",
          file=sys.stderr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)
    print(f"Loading model: {args.model}", file=sys.stderr)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map=device
        )
    except Exception as e:
        # Fallback to a fully open model if HF gating blocks Llama.
        print(f"Failed to load {args.model}: {e}\nFalling back to TinyLlama-1.1B-Chat-v1.0",
              file=sys.stderr)
        fb = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(fb)
        model = AutoModelForCausalLM.from_pretrained(
            fb, torch_dtype=torch.float16, device_map=device
        )

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    payload = {}
    for mode in sorted(df["mode"].unique()):
        df_m = df[df["mode"] == mode].reset_index(drop=True)
        print(f"  extracting mode={mode}  n={len(df_m)}", file=sys.stderr)
        embeds = extract_for_mode(df_m, model, tokenizer, device, args.Ns,
                                  batch_size=args.batch_size)
        for N, arr in embeds.items():
            payload[f"X__{mode}__N{N}"] = arr
        payload[f"y__{mode}"] = df_m["label"].to_numpy(dtype=object)
        payload[f"src__{mode}"] = df_m["source"].to_numpy(dtype=object)

    payload["Ns"] = np.array(args.Ns, dtype=np.int32)
    # Determine # layers from any shape.
    any_key = next(k for k in payload if k.startswith("X__"))
    num_layers = payload[any_key].shape[1]
    payload["layers"] = np.arange(num_layers, dtype=np.int32)
    payload["modes"] = np.array(sorted(df["mode"].unique()), dtype=object)
    np.savez_compressed(args.output, **payload)
    print(f"\nSaved embeddings -> {args.output}  (layers={num_layers}, Ns={args.Ns})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
