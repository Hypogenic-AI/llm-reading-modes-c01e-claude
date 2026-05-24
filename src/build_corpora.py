"""
Build the unified mode-detection corpus.

Produces data/chunks.parquet with columns:
    text:    raw text passage
    mode:    one of {authorship, ai_vs_human, register, typo}
    label:   string class label within the mode
    source:  origin tag (e.g., 'gutenberg:JA1', 'hc3:reddit_eli5')

Each mode contributes balanced classes. We oversample to a target per-class count.

Modes
-----
1) authorship   (Gutenberg, 11 classes)
2) ai_vs_human  (HC3, binary)
3) register     (LLM-generated formal vs. casual; built once via OpenAI API and cached)
4) typo         (synthetic QWERTY-adjacent vs. Dvorak-adjacent typo injection on clean text)
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
DATASETS = ROOT / "datasets"

SEED = 42
random.seed(SEED)

# Target ~600 samples per class so every mode contributes a comparable amount.
SAMPLES_PER_CLASS = 600
MIN_CHARS = 400   # passages shorter than this are skipped (need ~80+ tokens)
MAX_CHARS = 4000  # cap to avoid pathological outliers


# ---------------------------------------------------------------------------
# Mode 1 — Gutenberg authorship
# ---------------------------------------------------------------------------

GUTENBERG_LABELS = {
    "GE_Silas_Marner.txt": "GE",
    "GS_Sherlock.txt": "GS",
    "HM1_Moby_Dick.txt": "HM",
    "HM2_Typee.txt": "HM",
    "JA1_Pride_and_Prejudice.txt": "JA",
    "JA2_Sense_and_Sensibility.txt": "JA",
    "JA3_Emma.txt": "JA",
    "JJ_Ulysses.txt": "JJ",
    "MT1_Tom_Sawyer.txt": "MT",
    "MT2_Huckleberry_Finn.txt": "MT",
    "NH1_Scarlet_Letter.txt": "NH",
}


def _strip_gutenberg(text: str) -> str:
    """Remove the Gutenberg license / boilerplate to focus on the work."""
    start = re.search(r"\*\*\* ?START OF .* \*\*\*", text)
    end = re.search(r"\*\*\* ?END OF .* \*\*\*", text)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()]
    return text


def _split_into_passages(text: str, target_chars: int = 1200) -> list[str]:
    """Split text into ~target_chars-sized passages broken on paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", text)
    out, buf = [], ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 < target_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if len(buf) >= MIN_CHARS:
                out.append(buf[:MAX_CHARS])
            buf = p
    if len(buf) >= MIN_CHARS:
        out.append(buf[:MAX_CHARS])
    return out


def build_authorship() -> list[dict]:
    rows = []
    by_author: dict[str, list[tuple[str, str]]] = {}
    for fname, author in GUTENBERG_LABELS.items():
        path = DATASETS / "gutenberg" / fname
        text = _strip_gutenberg(path.read_text(encoding="utf-8", errors="ignore"))
        passages = _split_into_passages(text)
        by_author.setdefault(author, []).extend([(fname, p) for p in passages])

    # Down/up-sample per author to target.
    for author, passages in by_author.items():
        random.shuffle(passages)
        if len(passages) > SAMPLES_PER_CLASS:
            passages = passages[:SAMPLES_PER_CLASS]
        for src_file, p in passages:
            rows.append({
                "text": p,
                "mode": "authorship",
                "label": author,
                "source": f"gutenberg:{src_file}",
            })
    return rows


# ---------------------------------------------------------------------------
# Mode 2 — AI vs. Human (HC3)
# ---------------------------------------------------------------------------

def build_ai_vs_human() -> list[dict]:
    rows = []
    human, ai = [], []
    with open(DATASETS / "HC3" / "all.jsonl") as f:
        for line in f:
            r = json.loads(line)
            src = r.get("source", "unknown")
            for a in r.get("human_answers", []) or []:
                a = (a or "").strip()
                if len(a) >= MIN_CHARS:
                    human.append((a[:MAX_CHARS], src))
            for a in r.get("chatgpt_answers", []) or []:
                a = (a or "").strip()
                if len(a) >= MIN_CHARS:
                    ai.append((a[:MAX_CHARS], src))

    random.shuffle(human); random.shuffle(ai)
    n = min(len(human), len(ai), SAMPLES_PER_CLASS)
    for txt, src in human[:n]:
        rows.append({"text": txt, "mode": "ai_vs_human", "label": "human",
                     "source": f"hc3:{src}"})
    for txt, src in ai[:n]:
        rows.append({"text": txt, "mode": "ai_vs_human", "label": "ai",
                     "source": f"hc3:{src}"})
    return rows


# ---------------------------------------------------------------------------
# Mode 3 — Register (formal vs. casual)
#
# We use OpenAI's gpt-4.1-mini to generate paired formal & casual passages from
# the same content prompts. The pairs are cached to disk so repeated runs are
# cheap.
# ---------------------------------------------------------------------------

REGISTER_CACHE = DATA / "register_cache.jsonl"


def _gen_register_pairs(n_pairs: int = 250) -> list[dict]:
    """Generate n_pairs of (formal, casual) passages via OpenAI."""
    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover
        print("openai SDK missing; skipping register generation", file=sys.stderr)
        return []

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing; skipping register generation", file=sys.stderr)
        return []

    client = OpenAI()
    # Seed topics — short cues to drive content diversity.
    topics = [
        "the rise of cooperative board games", "how lightning forms in a storm",
        "why bread dough rises", "the history of the metric system",
        "the migration patterns of monarch butterflies", "how submarines stay submerged",
        "the discovery of penicillin", "why we yawn",
        "the origins of Halloween", "how passports came to exist",
        "the chemistry of sourdough fermentation", "how lighthouses are powered",
        "why some species hibernate", "the invention of the wheel",
        "how a microwave oven works", "the rules of cricket",
        "how vaccines train the immune system", "the history of jazz",
        "why volcanoes erupt", "how cats purr",
        "the principles of bicycle balance", "how a refrigerator works",
        "the geology of the Grand Canyon", "the history of denim jeans",
        "how chameleons change color", "the physics of rainbows",
        "why coffee gives energy", "the engineering of suspension bridges",
        "how plants photosynthesize", "the cultural history of coffee houses",
    ]
    random.shuffle(topics)
    if n_pairs > len(topics):
        # Sample with replacement; we ask for two different passages each time.
        topics = (topics * (1 + n_pairs // len(topics)))[:n_pairs]
    else:
        topics = topics[:n_pairs]

    out = []
    if REGISTER_CACHE.exists():
        for line in REGISTER_CACHE.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        print(f"register cache: loaded {len(out)} pairs", file=sys.stderr)

    have_topics = {row["topic"] for row in out}
    needed = [t for t in topics if t not in have_topics][: max(0, n_pairs - len(out))]
    print(f"register: generating {len(needed)} new pairs", file=sys.stderr)

    with REGISTER_CACHE.open("a") as fh:
        for i, topic in enumerate(needed):
            prompt = (
                f"Write two short essays on the topic: '{topic}'.\n\n"
                "Essay A must be FORMAL ACADEMIC: third-person, complex sentences, "
                "Latinate vocabulary, no contractions, citations-like hedging, ~250 words.\n\n"
                "Essay B must be CASUAL CONVERSATIONAL: first/second person, contractions, "
                "informal idioms, short sentences, exclamations, the way one would explain "
                "to a friend at a bar, ~250 words.\n\n"
                "Both essays must contain comparable factual content but differ ONLY in "
                "register and style.\n\n"
                "Output exactly in this format with no extra text:\n"
                "###FORMAL\n<essay A>\n###CASUAL\n<essay B>\n"
            )
            try:
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=1400,
                )
                content = resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                print(f"  api error on '{topic}': {e}", file=sys.stderr)
                continue
            try:
                _, rest = content.split("###FORMAL", 1)
                formal, casual = rest.split("###CASUAL", 1)
                formal = formal.strip()
                casual = casual.strip()
            except ValueError:
                print(f"  parse error on '{topic}', skipping", file=sys.stderr)
                continue
            if len(formal) < MIN_CHARS or len(casual) < MIN_CHARS:
                continue
            row = {"topic": topic, "formal": formal[:MAX_CHARS], "casual": casual[:MAX_CHARS]}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            out.append(row)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(needed)}", file=sys.stderr)

    return out


def build_register() -> list[dict]:
    rows = []
    pairs = _gen_register_pairs(n_pairs=min(SAMPLES_PER_CLASS, 280))
    for r in pairs:
        rows.append({"text": r["formal"], "mode": "register", "label": "formal",
                     "source": f"openai:formal:{r['topic'][:30]}"})
        rows.append({"text": r["casual"], "mode": "register", "label": "casual",
                     "source": f"openai:casual:{r['topic'][:30]}"})
    return rows


# ---------------------------------------------------------------------------
# Mode 4 — Typing-keyboard mode (QWERTY vs. Dvorak typo injection)
#
# Take clean text (Gutenberg passages), inject typos using QWERTY-adjacency
# or Dvorak-adjacency keyboards. The model never sees the keyboard layout
# label — we test whether the typo-substitution pattern is detectable from
# the text alone.
# ---------------------------------------------------------------------------

QWERTY_ADJ = {
    'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etfd', 't': 'ryfg', 'y': 'tugh',
    'u': 'yihj', 'i': 'uokj', 'o': 'iplk', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'sefcx', 'f': 'drgvc', 'g': 'fthbv',
    'h': 'gyjnb', 'j': 'hukmn', 'k': 'jilom', 'l': 'kop',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
    'n': 'bhjm', 'm': 'njk',
}

DVORAK_ADJ = {
    # Dvorak top row: , . p y f g c r l
    # home row:        a o e u i d h t n s
    # bottom row:      ; q j k x b m w v z
    'p': 'yf', 'y': 'pf', 'f': 'pgc', 'g': 'frc', 'c': 'grl', 'r': 'cl', 'l': 'r',
    'a': 'oeu', 'o': 'aeu', 'e': 'oui', 'u': 'eid', 'i': 'udh', 'd': 'iht',
    'h': 'dtn', 't': 'hns', 'n': 'ts', 's': 'n',
    'q': 'jk',  'j': 'qkx', 'k': 'jxb', 'x': 'kbm', 'b': 'xmw', 'm': 'bwv',
    'w': 'mvz', 'v': 'wz', 'z': 'v',
}


def _inject_typos(text: str, adj: dict, rate: float = 0.06) -> str:
    """Inject typos: with prob `rate`, substitute a character with a layout-adjacent one."""
    chars = list(text)
    rng = random.Random(hash(text) & 0xFFFFFFFF)
    for i, ch in enumerate(chars):
        low = ch.lower()
        if low in adj and rng.random() < rate:
            sub = rng.choice(adj[low])
            chars[i] = sub.upper() if ch.isupper() else sub
    return "".join(chars)


def build_typo() -> list[dict]:
    rows = []
    # Source text: use HC3 *human* answers (varied prose, not in Llama training repeatedly).
    sources: list[tuple[str, str]] = []
    with open(DATASETS / "HC3" / "all.jsonl") as f:
        for line in f:
            r = json.loads(line)
            src = r.get("source", "unknown")
            for a in r.get("human_answers", []) or []:
                a = (a or "").strip()
                if MIN_CHARS <= len(a) <= MAX_CHARS:
                    sources.append((a, src))
    random.shuffle(sources)
    sources = sources[: 2 * SAMPLES_PER_CLASS]

    half = len(sources) // 2
    qwerty_src = sources[:half]
    dvorak_src = sources[half:]

    for txt, src in qwerty_src:
        rows.append({"text": _inject_typos(txt, QWERTY_ADJ, rate=0.06),
                     "mode": "typo", "label": "qwerty",
                     "source": f"typo_qwerty:{src}"})
    for txt, src in dvorak_src:
        rows.append({"text": _inject_typos(txt, DVORAK_ADJ, rate=0.06),
                     "mode": "typo", "label": "dvorak",
                     "source": f"typo_dvorak:{src}"})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building corpora...", file=sys.stderr)
    parts = []

    print("  [authorship]", file=sys.stderr)
    parts.extend(build_authorship())
    print("  [ai_vs_human]", file=sys.stderr)
    parts.extend(build_ai_vs_human())
    print("  [register] (may call OpenAI API; ~3-5 minutes if cache empty)", file=sys.stderr)
    parts.extend(build_register())
    print("  [typo]", file=sys.stderr)
    parts.extend(build_typo())

    df = pd.DataFrame(parts)
    print("\nCorpus summary:")
    print(df.groupby(["mode", "label"]).size().to_string())
    print(f"\nTotal: {len(df)} passages")
    out = DATA / "chunks.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
