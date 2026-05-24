# Cloned Repositories

## Repo 1: DetectGPT

- **URL**: https://github.com/eric-mitchell/detect-gpt
- **Purpose**: Reference implementation of DetectGPT — zero-shot AI-text detection via
  probability curvature. Useful as a "machine vs. human mode" baseline that doesn't
  require training a classifier.
- **Location**: `code/detect-gpt/`
- **Key files**:
  - `run.py` — main experiment runner (DetectGPT + baselines)
  - `custom_datasets.py` — XSum, SQuAD, WritingPrompts, PubMedQA dataset loaders
  - `paper_scripts/` — shell scripts that reproduce paper figures
- **Requirements**: see `requirements.txt` (transformers, datasets, sentencepiece;
  expects GPU for any non-trivial scale)
- **Application to our research**: Provides a clean curvature-based mode detector we
  can adapt to compare against probing-classifier baselines.

## Repo 2: HC3 (ChatGPT-Comparison-Detection)

- **URL**: https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
- **Purpose**: HC3 dataset + reference detectors (GLTR, log-prob, perplexity-based,
  RoBERTa fine-tuned) for human vs. ChatGPT classification.
- **Location**: `code/chatgpt-comparison-detection/`
- **Key files**: detector demos and training notebooks
- **Application**: Provides baseline classifiers and feature pipelines we can adapt for
  multi-mode classification.

## Repo 3: M4

- **URL**: https://github.com/mbzuai-nlp/M4
- **Purpose**: Bundles the M4 multi-generator/multi-domain/multi-lingual MGT detection
  dataset and reference code.
- **Location**: `code/M4/` (~423 MB — most of it is the data under `code/M4/data/`)
- **Key files**:
  - `data/*.jsonl` — domain × generator pairs
  - `M4__Multidomain__Multimodel_and_Multilingual_Machine_Generated_Text_Detection.pdf` —
    the paper itself bundled in the repo
- **Application**: Largest cross-generator dataset on hand. Lets us test whether a
  "mode" probe trained on one (domain, generator) pair generalizes to others —
  directly relevant to the "isolate mode as a latent variable" question.

## Notes for experiment runner

- All three repos may need their own environment setup steps before use; **do not**
  pollute the workspace venv when installing repo-specific requirements.
- For just running probes against open-source LLMs (the Sarfati-style approach), only
  `transformers`, `torch`, and `scikit-learn` are needed — none of these three repos
  is strictly required, but they provide baselines + datasets.
- Recommended first experiment: replicate Sarfati's last-token probe on `datasets/gutenberg/`
  using Llama-3.2-1B (or similar). Then extend the same probe pipeline to:
  - Human vs. ChatGPT (HC3) → does the same probe family pick up "AI mode"?
  - Different ChatGPT versions (M4) → can we attribute to a specific generator?
  - Synthetic typo-injected text (constructed in workspace) → does the probe pick up
    typing patterns?
