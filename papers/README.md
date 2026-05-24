# Downloaded Papers

This directory contains papers gathered for the research project on "Modes" in LLM Reading.
Papers are organized chronologically and by topic relevance.

## Most Relevant (Top Priority)

### 1. Sarfati et al. 2025 — Literary Style in Prompt Embeddings
- **File**: `2025_Sarfati_LiteraryStyleInPromptEmbeddings.pdf`
- **arXiv**: 2505.17071
- **Authors**: Raphaël Sarfati, Haley Moller, Toni J. B. Liu, Nicolas Boullé, Christopher Earls
- **Key Contribution**: Shows LLM deep embeddings encode literary STYLE (an intangible attribute),
  not just factual content. Uses Llama-3.2-1B; trains linear (SVM) and MLP probes on the last
  embedding of short (8-128 token) excerpts to classify which novel/author they come from.
- **Why critical**: This is the closest existing work to our hypothesis. The hypothesis asks
  what "modes" LLMs detect — Sarfati shows that author/style is one such latent variable
  encoded in deep representations. Provides a direct methodological template: tokenize → chunk →
  last-token embedding per layer → probe classifier.
- **Key findings**:
  - Style separability emerges deep in the model (later transformer layers, longer contexts).
  - Inter-author confusion > intra-author confusion in multiclass probe → style is the signal.
  - Style lives in the first ~16 PCA dimensions of the embedding.
  - Shuffling tokens preserves separability → signal is more lexical than syntactic.
  - Style geometry survives translation (French ↔ English probe transfer works).

### 2. Mitchell et al. 2023 — DetectGPT
- **File**: `2023_Mitchell_DetectGPT.pdf`
- **arXiv**: 2301.11305
- **Authors**: Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, Chelsea Finn
- **Key Contribution**: Zero-shot AI-text detection via log-probability curvature.
  Machine-generated text sits in negative-curvature regions of pθ; human text does not.
- **Method**: Sample k perturbations x̃ᵢ via T5, compare logpθ(x) − mean(logpθ(x̃ᵢ)).
- **Why relevant**: Treats "machine vs. human" as a detectable mode without supervised training.
  Establishes baseline AUROC on news/Wikipedia/XSum: GPT-2 (0.99), GPT-Neo-2.7B (0.95+).
- **Code**: https://github.com/eric-mitchell/detect-gpt (cloned to `code/detect-gpt/`)

### 3. Rivera Soto et al. 2021 — Universal Authorship Representations
- **File**: `2021_RiveraSoto_UniversalAuthorshipRep.pdf`
- **arXiv**: 2109.07020
- **Key Contribution**: Learns embeddings for authorship verification across Amazon reviews,
  fanfiction, and Reddit. Tests cross-domain transfer of style representations.
- **Why relevant**: Shows authorship/style is learnable from text alone, but transfer
  across domains is uneven — suggests style ≠ purely surface artifacts.

### 4. Alain & Bengio 2016 — Linear Classifier Probes
- **File**: `2016_Alain_LinearProbes.pdf`
- **arXiv**: 1610.01644
- **Key Contribution**: Foundational paper for the probing classifier methodology used in
  Sarfati and most interpretability research. Train a linear classifier on frozen activations
  to measure what information each layer encodes.
- **Why relevant**: Core methodology for any "isolate mode as latent variable" experiment.

### 5. Li et al. 2022 — Emergent World Representations (Othello-GPT)
- **File**: `2022_Li_EmergentWorldRepresentations.pdf`
- **arXiv**: 2210.13382
- **Key Contribution**: Probing classifiers find that a GPT trained only on Othello move
  sequences develops an internal representation of board state, manipulable via intervention.
- **Why relevant**: Demonstrates that latent variables (here: world state) can be both
  decoded AND causally controlled — the "isolate mode as a latent variable" half of our
  hypothesis.

## AI Text Detection Suite

### 6. Ippolito et al. 2019 — Automatic Detection is Easiest when Humans are Fooled
- **File**: `2019_Ippolito_AutomaticDetection.pdf`
- **Key**: Shows trade-off between human-fool quality and machine-detectability across
  decoding strategies (top-k, nucleus, untruncated). Statistical regularities differ
  even when text looks human.

### 7. Chakraborty et al. 2023 — Counter Turing Test (CT²) & ADI
- **File**: `2023_Chakraborty_CounterTuringTest.pdf`
- **Key**: Comprehensive benchmark of AI-text-detection methods; introduces AI Detectability
  Index. Larger LLMs are less detectable.

### 8. Roy et al. 2025 — Comprehensive Dataset for Human vs. AI Text Detection
- **File**: `2025_Roy_ComprehensiveDataset.pdf`
- **Key**: 58K samples from NYT human + Gemma-2-9B, Mistral-7B, Qwen-2-72B, LLaMA-8B,
  Yi-Large, GPT-4o. Dataset on HuggingFace: gsingh1-py/train.

### 9. Zeng et al. 2025 — Human Texts are Outliers (OOD Detection)
- **File**: `2025_Zeng_HumanTextsAreOutliers.pdf`
- **Key**: Reframes AI-text detection as OOD: AI texts are ID, human texts are diverse
  outliers. Reaches 98.3% AUROC on DeepFake.

### 10. Schäfer & Steinebach 2025 — Stylistic & Statistical Feature Modeling
- **File**: `2025_Schaefer_StylisticFeatureModeling.pdf`
- **Key**: Uses 220 LFTK stylistic features (Kuperman age, lexical variation) to detect
  AI text. F1 90%+. Shows AI-generated text has higher word complexity and lower lexical
  variation than human text.

### 11. Thorat & Caines 2025 — DACTYL Adversarial Corpus
- **File**: `2025_Thorat_DACTYL.pdf`
- **Key**: Diverse adversarial AI-text corpus including few-shot/one-shot generations.
  Most existing detectors struggle.

### 12. Fu et al. 2025 — FDLLM Fingerprinting
- **File**: `2025_Fu_FDLLM_Fingerprinting.pdf`
- **Key**: Identifies WHICH LLM generated a piece of text (20 major LLMs, 90K samples).
  LoRA-adapted detector. Multilingual. 95% accuracy on new models.

### 13. Verma et al. 2023 — Ghostbuster
- **File**: `2023_Verma_Ghostbuster.pdf`
- **Key**: Uses weaker language models' probabilities as features. SOTA across domains.

### 14. Miralles-González et al. 2025 — PAWN
- **File**: `2025_MirallesGonzalez_PAWN.pdf`
- **Key**: Attention-weighted aggregation of next-token distribution metrics. Robust to
  domain shift and adversarial attacks.

## Probing / Interpretability

### 15. Amini & Ciaramita 2023 — In-Context Probing
- **File**: `2023_Amini_InContextProbing.pdf`
- **Key**: Probe contextualized representations under instruction prefix. More robust
  than fine-tuning or naive in-context learning.

### 16. Templeton et al. 2024 — Scaling Monosemanticity (Sparse Autoencoders)
- **File**: `2024_SparseAutoencoder.pdf`
- **Key**: SAEs on Claude 3 Sonnet extract interpretable features (cities, emotions,
  code patterns). Method for finding latent variables without supervision.

## Style / Authorship

### 17. Toshevska & Gievska 2025 — LLM-Based Text Style Transfer
- **File**: `2024_Toshevska_StyleTransferLLM.pdf`
- **Key**: Survey of LLM style transfer (politeness, formality, sentiment).

### 18. Fisher et al. 2024 — JAMDEC (Authorship Obfuscation)
- **File**: `2024_Fisher_JAMDEC_AuthorshipObfuscation.pdf`
- **Key**: Reverse problem — obfuscating author style at decode time using constrained
  decoding over GPT2-XL. Demonstrates style is something models can actively control.

## Search Strategy Summary

Paper-finder service ran six diligent searches across:
- AI-generated text detection
- Probing / latent variables / linear classifiers
- Authorship attribution and stylometry
- Speech transcription vs written text differences
- Machine-generated text detection methods
- Demographic / typing-error / keyboard-influenced text
- Sparse autoencoders for interpretability

Twelve plus four follow-up papers were downloaded based on relevance scores ≥ 1 and
direct alignment with the research hypothesis. See `paper_search_results/` directory
for raw JSONL search dumps.
