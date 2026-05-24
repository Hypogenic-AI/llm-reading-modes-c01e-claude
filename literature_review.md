# Literature Review: "Modes" in LLM Reading

**Research hypothesis**: LLMs detect various "modes" in text — dictation, AI-authored,
keyboard-layout influence, etc. — and these "modes" can potentially be isolated as
latent variables in LLM representations.

## 1. Research Area Overview

The hypothesis sits at the intersection of three established literatures:

1. **Machine-generated text detection** — a large, fast-moving field that defines and
   detects the "AI-authored" mode as a binary or multi-class classification problem.
2. **Authorship attribution and stylometry** — a much older field that treats authorial
   "style" as a hidden variable detectable from surface text.
3. **Mechanistic interpretability / probing** — a methodological field providing the
   tools (linear probes, SAEs, intervention) to ask "what does the model represent?"

The novelty of this project is to **unify (1) and (2) as instances of a more general
"mode" detection problem**, and to use (3) to test whether such modes can be cleanly
isolated as latent variables. The closest existing work to that synthesis is
**Sarfati et al. 2025**, which shows that LLM deep embeddings already encode literary
style as a near-linear subspace.

## 2. Key Papers (Ranked by Relevance)

### Sarfati et al. 2025 — Literary style in prompt embeddings (arXiv:2505.17071)
**Methodology** (most directly applicable to our hypothesis):
- Tokenize a literary work, split into N-token chunks (N ∈ {8, 16, …, 128}).
- For each chunk, pass through Llama-3.2-1B (16 layers).
- Collect the *last token's* embedding after each layer L.
- Train an SVM (binary) or MLP (multiclass) probe to classify chunks by source book.
- Vary L (depth) and N (context) to see when "style" emerges.

**Findings**:
- Binary book-vs-book separability climbs from ~50% at L=2 to >90% at L=10+, N≥64.
- Multiclass (13 books) MLP probe reaches ~75% accuracy with N=128, L=16.
- Same-author confusion is markedly higher than across-author → style ≠ topic.
- Style lies within ~16 PCA dimensions (intrinsic dim. ~20).
- **Token shuffling barely hurts probe accuracy** → style signal is lexical, not syntactic.
- Probes trained on French author-pairs transfer to English translations of the same
  pairs → style geometry is partially language-independent.

**Why this is the load-bearing paper**: Sarfati provides both (i) evidence that an
intangible attribute (style) is encoded as a latent variable, and (ii) a complete
methodology we can directly extend to other "modes" (dictation, AI-authorship, keyboard
artifacts).

### Mitchell et al. 2023 — DetectGPT (arXiv:2301.11305)
**Methodology**: Zero-shot. To test if text x came from model pθ:
1. Generate k perturbations x̃ᵢ using T5 mask-filling.
2. Score d(x) = log pθ(x) − mean(log pθ(x̃ᵢ)).
3. Threshold: large positive d ⇒ machine-generated (negative curvature region).

**Findings**: AUROC 0.95+ on GPT-Neo-20B-generated news, beating strongest zero-shot
baseline by 0.14. Works because LLM samples lie in negative-curvature regions of pθ;
human text does not.

**Why relevant**: Establishes that "AI vs. human" mode is detectable *without* learning
a classifier — model probabilities alone reveal it. Suggests "mode" may be detectable
at multiple levels: (a) hidden-state representations (Sarfati), (b) output probabilities
(DetectGPT), (c) statistical text features (Schäfer & Steinebach 2025).

### Alain & Bengio 2016 — Linear classifier probes (arXiv:1610.01644)
The methodological grandparent. Train a linear classifier on frozen activations at each
layer. Layer-wise probe accuracy reveals where information is encoded. Foundational for
all interpretability work below.

### Li et al. 2022 — Othello-GPT emergent world representations (arXiv:2210.13382)
Proves the strong form of the latent-variable claim: a transformer trained only on
Othello move sequences develops a probe-decodable, *causally interveneable* internal
board state. This is the gold standard for "isolate mode as latent variable" — it
combines decoding (probe) with control (intervention).

### Rivera Soto et al. 2021 — Universal authorship representations (arXiv:2109.07020)
Cross-domain transfer of style/authorship embeddings (Amazon reviews, fanfiction, Reddit).
Some domain pairs transfer well; others don't. Implication: style is partially separable
from domain/topic, but the separation is incomplete.

### Templeton et al. 2024 — Scaling monosemanticity (Sparse Autoencoders)
SAEs on Claude 3 Sonnet extract interpretable features — cities, emotions, deception,
code patterns. Most-importantly, this method is **unsupervised** — it discovers candidate
"modes" without needing them labeled in advance. Highly relevant to the hypothesis's
second half: "What *other* modes are LLMs detecting?" — SAEs are the natural tool.

### Schäfer & Steinebach 2025 — Stylistic & statistical feature modeling
Hand-crafted features (220 from LFTK) classify AI vs. human at F1 90%+. Key insight:
AI text has higher Kuperman age (word complexity) and lower lexical variation than
human text. Provides a non-neural baseline that any probing study should beat.

### Other AI-detection papers (Chakraborty 2023, Ippolito 2019, Verma 2023, Zeng 2025,
Thorat 2025, Roy 2025, Miralles-González 2025, Fu 2025)
Collectively establish:
- Detection AUROC declines as LLM size grows (larger LLMs less detectable).
- Detection is harder cross-domain and cross-generator.
- Adversarial paraphrasing and few-shot generation evade many detectors.
- LLM fingerprinting (which-LLM-generated-this) is feasible at >95% accuracy on
  in-distribution data.

## 3. Methodologies Common in the Literature

| Method | Used by | Use for "modes" research |
|--------|---------|---------------------------|
| **Linear / MLP probing on frozen activations** | Alain & Bengio; Sarfati; Li (Othello); Marks & Tegmark | First-line tool to test "is mode X represented?" |
| **Layer-wise scan** (probe at every layer) | Sarfati; Alain & Bengio | Locates *where* in the model a mode emerges |
| **PCA / SVD on activations** | Sarfati; Park et al. | Reveals dimensionality of the mode subspace |
| **Causal intervention / activation steering** | Li (Othello); Templeton; Gu et al. 2025 | Tests whether the mode is *load-bearing* for behavior |
| **Output-distribution-based detection** | Mitchell (DetectGPT); Ippolito; Verma (Ghostbuster) | Zero-shot baselines, no probe training needed |
| **Hand-crafted feature classifiers** | Schäfer; Sharma & Mansuri (SemEval-2024) | Strong non-neural baselines |
| **Sparse autoencoders / dictionary learning** | Templeton; Bricken et al. 2023 | Unsupervised discovery of latent modes |
| **Cross-domain / cross-generator transfer** | Rivera Soto; M4 benchmark | Tests whether a mode is intrinsic vs. surface |
| **Token shuffling ablation** | Sarfati; Viswanathan 2025 | Distinguishes lexical from syntactic signal |

## 4. Standard Baselines for "Mode" Detection Experiments

- **Hand-crafted stylistic features + Logistic Regression / SVM** — Schäfer-style. Cheap, robust.
- **Mean per-token log-probability** under the source LLM (Solaiman 2019).
- **DetectGPT** — perturbation-based curvature.
- **Fine-tuned RoBERTa / DistilBERT classifier** (e.g., HC3 reference detector).
- **Linear probe on Llama-3.2-1B last-token embeddings** (Sarfati replication).

## 5. Evaluation Metrics

- **AUROC** — primary metric for binary detection (most papers).
- **Accuracy** — for closed-set multiclass attribution (Sarfati uses this).
- **F1 / Macro-F1** — when classes are imbalanced.
- **FPR @ low TPR** (e.g., FPR95) — used in OOD-detection framings (Zeng 2025).
- **Per-class confusion** — important for revealing intra-mode confusion patterns
  (Sarfati's intra/extra-author analysis).

## 6. Datasets in the Literature

| Dataset | Used by | Description |
|--------|---------|-------------|
| Project Gutenberg | Sarfati | Public-domain literature for author-attribution |
| HC3 | Guo 2023, many | 24K Q&A with human + ChatGPT answers |
| M4 | Wang 2024 | Multi-generator × multi-domain × multi-lingual |
| Comprehensive (Roy 2025) | Roy 2025 | NYT + 6 LLMs |
| DACTYL | Thorat 2025 | Few-shot adversarial AI text |
| FD-Dataset | Fu 2025 | 90K samples from 20 LLMs (fingerprinting) |
| GPT-3-generated XSum/SQuAD/WritingPrompts | DetectGPT | Standard zero-shot detection eval |

For "less-canonical" modes (dictation, typing artifacts), no off-the-shelf labeled
dataset exists — we'd need to construct synthetic data (e.g., LibriSpeech transcripts,
typo-injection scripts).

## 7. Gaps and Opportunities Relevant to the Hypothesis

1. **"Modes" beyond AI/human and author/style are largely unexplored.** Almost the
   entire literature treats these two as separate problems with separate detectors.
   No prior work asks: are dictation, typing-keyboard-layout, register, etc., all
   instances of the same "mode" abstraction? This is the project's main novelty.

2. **Unsupervised discovery of modes** (via SAEs / dictionary learning) has been
   demonstrated for features like "code", "emotion", "deception" but not systematically
   for stylistic or production-channel modes.

3. **Sarfati shows style is encoded but does not intervene.** A causal-style experiment
   (steer the model along the "literary style" direction and observe output change)
   would strengthen "mode as latent variable" beyond decoding.

4. **Cross-mode interference is unstudied.** If we train a probe to detect dictation,
   does it confuse with "informal register"? With "AI generation"? Disentangling
   modes is itself an open problem.

5. **Robustness to natural distribution shift** (cross-domain, cross-generator,
   adversarial) is the main failure mode of all AI-text detectors. Mode probes
   trained on hidden states (Sarfati) may have better generalization than output-
   distribution detectors — worth testing.

## 8. Recommendations for Experimental Design

### Primary experiment (replicates Sarfati, extends to "AI vs. human" mode)

**Setup**: Llama-3.2-1B (free, fast, 16 layers).

**Probe pipeline**:
1. For each text sample, tokenize and take only the first N tokens (N = 64 or 128).
2. Forward pass through the model; collect the last-token hidden state at every layer L.
3. Train an SVM (binary) or MLP (multiclass) probe on these embeddings.
4. Evaluate accuracy by (N, L) — a heatmap as in Sarfati Fig. 2B.

**Modes to probe**:
- **Mode 1: Authorship** — Project Gutenberg corpus (11 novels, 6 authors).
  Expected: replicate Sarfati's ~75% multiclass / ~95% binary at N=128, L=16.
- **Mode 2: Human vs. ChatGPT** — HC3 dataset.
  Open question: at what layer L does AI-mode become linearly decodable?
  Compare with DetectGPT AUROC as zero-shot baseline.
- **Mode 3: Generator attribution** — M4 dataset (which LLM generated this?).
  Test cross-domain generalization (train on arxiv_chatGPT, test on reddit_chatGPT).

### Secondary experiments (novelty / hypothesis-extending)

- **Synthetic typing-error mode**: Inject QWERTY-adjacent typos into clean text at
  varying rates. Train a probe to detect typo-injection. Compare to a probe trained
  to detect actually-mistyped Wikipedia revision-history data (if available).
- **Probe transfer across modes**: Does a probe trained on (author = JA vs. MT)
  generalize at all to (formal vs. ELI5)? Same human authors, different "register"
  mode. Quantifies how factorized the mode subspaces really are.
- **PCA dimensionality scan**: Apply Sarfati's PCA technique to each mode subspace
  and report intrinsic dimension. Hypothesis: production-channel modes (dictation,
  typing) have lower-dim subspaces than authorship.

### Tertiary / stretch experiments

- **SAE feature interpretation**: Train a sparse autoencoder on Llama-3.2-1B
  activations from a mixed-mode corpus. Inspect top-activating examples for each
  feature — do we discover unsupervised "mode" features?
- **Causal intervention**: Use linear probe weights as steering vectors. Project them
  out of the residual stream; observe how generations change. Tests whether modes
  are merely decodable or also load-bearing.

## 9. Recommended Resources Summary

- **Models**: Llama-3.2-1B (Sarfati replication), GPT-Neo / Pythia (compute-light alternatives)
- **Datasets**: Project Gutenberg (downloaded), HC3 (downloaded), M4 (in code/M4/data/)
- **Code baselines**: detect-gpt (zero-shot), HC3 reference detectors, M4 baselines
- **Probe library**: scikit-learn (SVM, MLP, LogReg); for SAEs: SAELens or train custom

The combination of (a) Sarfati's methodology, (b) the gathered datasets, and
(c) the cloned baselines is sufficient to run all of the primary and secondary
experiments above without further resource gathering.
