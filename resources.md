# Resources Catalog

This document catalogs all resources gathered for the "Modes" in LLM Reading project.

## Summary

- **Papers**: 18 PDFs downloaded covering AI-text detection, probing, authorship attribution,
  and mechanistic interpretability.
- **Datasets**: 3 datasets gathered (Project Gutenberg literary corpus, HC3 human-ChatGPT
  corpus, M4 multi-generator corpus bundled with code/M4).
- **Code**: 3 repositories cloned (DetectGPT, HC3 detectors, M4 toolkit).

## Papers

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| What's in a prompt? Literary style in prompt embeddings | Sarfati et al. | 2025 | `papers/2025_Sarfati_LiteraryStyleInPromptEmbeddings.pdf` | **Primary methodological template** — last-token probing on Llama-3.2-1B reveals style as latent variable |
| DetectGPT: Zero-shot probability-curvature detection | Mitchell et al. | 2023 | `papers/2023_Mitchell_DetectGPT.pdf` | Zero-shot AI-text baseline. Code cloned. |
| Universal Authorship Representations | Rivera Soto et al. | 2021 | `papers/2021_RiveraSoto_UniversalAuthorshipRep.pdf` | Cross-domain author embeddings |
| Understanding intermediate layers using linear classifier probes | Alain & Bengio | 2016 | `papers/2016_Alain_LinearProbes.pdf` | Foundational probing methodology |
| Emergent World Representations (Othello-GPT) | Li et al. | 2022 | `papers/2022_Li_EmergentWorldRepresentations.pdf` | Causal probe-intervention paradigm |
| Counter Turing Test (CT² + ADI) | Chakraborty et al. | 2023 | `papers/2023_Chakraborty_CounterTuringTest.pdf` | AI-text-detection benchmark |
| Comprehensive Dataset for Human vs. AI Detection | Roy et al. | 2025 | `papers/2025_Roy_ComprehensiveDataset.pdf` | NYT + 6 LLMs benchmark |
| Human Texts Are Outliers (OOD detection) | Zeng et al. | 2025 | `papers/2025_Zeng_HumanTextsAreOutliers.pdf` | One-class learning framing |
| Stylistic & statistical feature modeling | Schäfer & Steinebach | 2025 | `papers/2025_Schaefer_StylisticFeatureModeling.pdf` | 220 LFTK features, F1 90%+ |
| DACTYL adversarial corpus | Thorat & Caines | 2025 | `papers/2025_Thorat_DACTYL.pdf` | Few-shot adversarial AI text |
| FDLLM black-box LLM fingerprinting | Fu et al. | 2025 | `papers/2025_Fu_FDLLM_Fingerprinting.pdf` | Which-LLM-generated-this at 95% acc |
| Automatic detection easiest when humans fooled | Ippolito et al. | 2019 | `papers/2019_Ippolito_AutomaticDetection.pdf` | Sampling-strategy effects on detectability |
| LLM-based text style transfer survey | Toshevska & Gievska | 2025 | `papers/2024_Toshevska_StyleTransferLLM.pdf` | Style transfer landscape |
| JAMDEC authorship obfuscation | Fisher et al. | 2024 | `papers/2024_Fisher_JAMDEC_AuthorshipObfuscation.pdf` | Reverse problem — controllable style |
| PAWN — perplexity-attention-weighted networks | Miralles-González et al. | 2025 | `papers/2025_MirallesGonzalez_PAWN.pdf` | Robust AI-text detector |
| Ghostbuster | Verma et al. | 2023 | `papers/2023_Verma_Ghostbuster.pdf` | Weaker-LM-probability features |
| In-Context Probing | Amini & Ciaramita | 2023 | `papers/2023_Amini_InContextProbing.pdf` | Instruction-conditioned probing |
| Scaling Monosemanticity (Sparse Autoencoders) | Templeton et al. | 2024 | `papers/2024_SparseAutoencoder.pdf` | Unsupervised feature discovery |

Detailed per-paper summaries in `papers/README.md`. Raw paper-finder JSONL search outputs in
`paper_search_results/`.

## Datasets

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Project Gutenberg corpus | gutenberg.org | 11 books, ~7 MB | Authorship attribution | `datasets/gutenberg/` | Public domain. Replicates Sarfati setup. |
| HC3 (Human-ChatGPT Comparison) | HF: Hello-SimpleAI/HC3 | 24K Q&A, ~140 MB | AI vs. human detection | `datasets/HC3/` | CC-BY-SA. 5 domains. |
| M4 multi-generator corpus | github.com/mbzuai-nlp/M4 | ~416 MB | Multi-generator + multi-domain detection | `code/M4/data/` | Bundled with cloned repo. |
| (Optional) Roy 2025 NYT + 6 LLMs | HF: gsingh1-py/train | 7K rows, 161 MB | Human vs. AI; LLM attribution | Not downloaded | Available via `datasets.load_dataset('gsingh1-py/train')` |

Detailed loading instructions and download scripts in `datasets/README.md`.

## Code Repositories

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| DetectGPT | github.com/eric-mitchell/detect-gpt | Probability-curvature AI-text detection | `code/detect-gpt/` | Reference implementation |
| HC3 detectors | github.com/Hello-SimpleAI/chatgpt-comparison-detection | HC3 dataset + baseline classifiers | `code/chatgpt-comparison-detection/` | Includes GLTR, log-prob, RoBERTa baselines |
| M4 toolkit | github.com/mbzuai-nlp/M4 | Multi-generator MGT benchmark | `code/M4/` | Bundles dataset (416 MB) |

Details in `code/README.md`.

## Resource Gathering Notes

### Search Strategy

Ran nine paper-finder searches in increasing specificity:
1. "LLM detection of AI-generated text style mode"
2. "LLM probing latent style author dictation transcription"
3. "probing classifier latent variable language model representations"
4. "authorship attribution stylometry neural network"
5. "speech transcription text written language differences register"
6. "machine generated text detection GPT classifier features"
7. "typing keyboard error pattern detection text"
8. "sparse autoencoder feature interpretability large language model"
9. "DetectGPT zero-shot machine generated text"
10. "demographic prediction text writing style LLM"
11. "linear probing classifier hidden state language model"

This breadth covered each "mode" mentioned in the hypothesis (AI-authored, dictated,
keyboard-typed, demographic) plus the interpretability tooling we'll need.

### Selection Criteria

- Direct relevance to hypothesis (style/authorship/AI-detection/probing).
- Methodological transferability (i.e., can we copy/adapt the experimental setup).
- Recent (2023–2025 preferred) for state-of-the-art, plus a few foundational older papers
  (Alain & Bengio 2016; Ippolito 2019).
- Mix of empirical detection papers and mechanistic-interpretability papers.

### Challenges Encountered

1. **Wrong arXiv ID initially used for Sarfati paper** — the paper-finder returned a
   Semantic Scholar URL without arXiv ID; the obvious ID guess was wrong. Resolved
   via WebSearch.
2. **HuggingFace datasets script-based loading deprecated** — HC3 required direct
   JSONL download instead of `load_dataset()`.
3. **No widely available dataset for dictation/typing-keyboard modes** — only candidate
   sources are biometric keystroke datasets that don't pair with written-style equivalents.
   Synthetic data construction will be needed.

### Gaps and Workarounds

| Resource needed | Available? | Workaround |
|-----------------|------------|------------|
| Style/author classification dataset | Yes (Gutenberg) | n/a |
| AI vs. human text dataset | Yes (HC3, M4, Roy) | n/a |
| LLM attribution dataset | Yes (M4, FDLLM via paper) | n/a |
| Dictation vs. written text dataset | No | Construct from LibriSpeech transcripts + matched written paragraphs |
| Keyboard-layout-typo dataset | No | Synthetic: QWERTY-adjacent typo injection |
| Cross-mode benchmark | No | Construct by combining the above |

## Recommendations for Experiment Design

1. **Primary dataset(s)**:
   - Project Gutenberg (downloaded) for authorship-mode probing replication of Sarfati.
   - HC3 (downloaded) for AI-vs-human-mode probing.
   - M4 (in code/M4/data/) for cross-generator and cross-domain transfer.

2. **Primary baselines**:
   - Linear / MLP probe on frozen Llama-3.2-1B last-token embeddings (Sarfati replication).
   - DetectGPT zero-shot curvature detector (Mitchell — for AI-vs-human only).
   - LFTK stylistic-feature LogReg classifier (Schäfer — non-neural baseline).

3. **Primary metrics**:
   - Binary AUROC, multiclass accuracy, per-class confusion matrix.
   - Probe accuracy as function of layer depth L and context length N (Sarfati Fig. 2B style).
   - Intra- vs. extra-mode confusion ratios (analog of Sarfati's intra-author / extra-author).

4. **Methodological considerations**:
   - Strip Gutenberg license headers before tokenizing.
   - Watch chunk-boundary artifacts (Sarfati does not align chunks to syntactic units —
     replicate that to match results).
   - Use the same model (Llama-3.2-1B) across modes so that activation spaces are comparable.
   - When testing transfer/intervention, ensure train/test splits are by *book* (or by
     *question*), not by random shuffle of sentences — leakage will inflate scores.

5. **Code to adapt/reuse**:
   - `code/detect-gpt/run.py` — adapt as the "probability-based mode detector" baseline.
   - `code/chatgpt-comparison-detection/` — for HC3-specific feature pipelines.
   - Most experiments can be implemented in <500 lines using transformers + scikit-learn.
     No heavy framework dependency on the cloned repos.
