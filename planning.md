# Research Plan: "Modes" in LLM Reading

## Motivation & Novelty Assessment

### Why This Research Matters

LLMs trained on naturalistic web text appear sensitive to many subtle "modes" of
text production — whether a passage was dictated rather than typed, written by another
LLM, or marked by the keyboard layout the author used. If these signals are real,
they raise immediate questions for evaluation (benchmark contamination by AI text),
privacy (demographic / device inference from style), and steering (can we control
production modes deliberately?). Yet the literature has fragmented this work into
three silos: AI-text detection, authorship attribution, and probing for individual
features. No prior work tests whether these are instances of a single more general
**"mode" abstraction** that an LLM represents internally.

### Gap in Existing Work

Per `literature_review.md`:
- AI-vs-human detection (Mitchell DetectGPT, Verma Ghostbuster, Schäfer features)
  and authorship attribution (Rivera Soto, Sarfati) develop distinct methods for
  what may be the same problem.
- Sarfati 2025 shows literary style is encoded as a near-linear subspace in
  Llama-3.2-1B, but only tests one mode (authorship).
- Cross-mode interference — does an "AI-text" probe pick up on register, dictation,
  or typing artifacts? — is **unstudied**.
- "Modes" beyond authorship and AI-generation (register, dictation, keyboard layout)
  are not probed at all in prior work.

### Our Novel Contribution

We treat **mode as a unified latent variable** and use a single probing pipeline
(Sarfati-style linear/MLP probes on Llama-3.2-1B last-token embeddings) to test:

1. Whether multiple distinct mode types — authorship, AI-generation, register,
   keyboard-layout typo pattern — are all linearly decodable from the same hidden
   states.
2. Whether mode subspaces are **disentangled** (orthogonal) or overlapping, by
   measuring cross-mode probe transfer and inter-direction cosine similarity.
3. Whether the **intrinsic dimensionality** of each mode subspace differs in ways
   that reveal structure (e.g., production-channel modes simpler than authorial
   modes).
4. Whether mode probes generalize to **out-of-distribution surface forms** of the
   same mode (a probe trained on QWERTY typos applied to mistyped Wikipedia text).

This unification — testing many modes within a single representation space and
measuring their geometric relationships — is novel.

### Experiment Justification

| # | Experiment | Why it's needed |
|---|------------|-----------------|
| 1 | Replicate Sarfati on Gutenberg authorship | Verify our methodology produces the published baseline before extending; sanity-check pipeline |
| 2 | Probe AI vs. human (HC3) layer-wise | First test of whether the *same* probe family that detects authorship also detects AI-mode, and where in the model it emerges |
| 3 | Probe novel synthetic modes (QWERTY typos, dictation-style disfluency, register) | Test whether modes *not in any training set* are nonetheless represented — central novelty claim |
| 4 | Cross-mode transfer & subspace geometry | Test the unification claim: are mode subspaces orthogonal? do probes confuse modes? |

## Research Question

Are diverse "modes" of text production — authorship, AI vs. human generation,
register, and keyboard-layout-induced typing artifacts — represented as **linearly
decodable, geometrically disentangled latent variables** in the hidden states of a
mid-size LLM (Llama-3.2-1B)?

## Hypothesis Decomposition

- **H1 (Decodability).** A linear probe on Llama-3.2-1B last-token embeddings
  achieves >75% accuracy on each of: authorship (Gutenberg 11-class), AI vs. human
  (HC3), register (formal vs. casual), and typo-pattern (QWERTY vs. Dvorak-adjacent).
- **H2 (Depth.** Mode accuracy increases with layer depth from L=2 to L≈10, then
  plateaus or declines slightly (replicating Sarfati Fig. 2B shape).
- **H3 (Low-dimensional subspace).** Each mode subspace, as recovered by PCA on the
  probe-decision space, has intrinsic dimensionality ≲ 20.
- **H4 (Partial disentanglement).** Mode subspaces are not orthogonal (cross-mode
  cosine > 0), but cross-mode probe transfer is well below within-mode (transfer
  accuracy < 0.7 × within accuracy).
- **H5 (Within-mode generalization).** A probe trained on synthetic QWERTY typos
  generalizes (above chance) to natural-text typo distributions not seen in training.

## Proposed Methodology

### Approach

Single-pipeline probing study, following Sarfati 2025:

1. Tokenize text → chunks of length N (varying N ∈ {16, 32, 64, 128}).
2. Forward pass through Llama-3.2-1B; collect **last-token hidden states** at each
   of 16 layers.
3. For each (mode, N, L), train a probe (LogReg / MLP) and measure held-out accuracy.
4. Repeat for 4 modes within the same embedding cache; analyze geometry.

### Experimental Steps

**Phase A — Pipeline setup**
1. Download Llama-3.2-1B; verify forward-pass returns `hidden_states` for all layers.
2. Implement chunked tokenization + last-token extraction → embeddings cache.

**Phase B — Build mode corpora** (all balanced ~1000 samples per class)
1. **Authorship**: Gutenberg, 11-class (already labeled).
2. **AI-vs-human**: HC3 paired answers (binary).
3. **Register**: GPT-4-class LLM generates a passage in {formal-academic, casual-conversational} for each of ~500 prompts (binary).
4. **Typo pattern**: Take clean text, inject typos using QWERTY-adjacency vs.
   Dvorak-adjacency models (binary). Held-out: natural Wikipedia revision noise
   (if obtainable) or text with random non-adjacent typos.

**Phase C — Run probes** (LogReg + MLP) at every L for N ∈ {16, 64, 128}.

**Phase D — Geometric analysis**
- PCA on per-class mean embeddings to recover mode subspace; report explained-variance dimensionality.
- Cosine similarity matrix between mode directions (first PC of each mode's class-mean differences).
- Cross-mode probe transfer matrix: train probe on mode A, test on mode B examples
  using A's labels-by-analog.

### Baselines

- **Random chance** per task.
- **Bag-of-words TF-IDF + LogReg**: a surface-feature baseline. If neural probes
  don't beat this, hidden-state representation adds no signal.
- **Last-layer only** (cf. probes at every L): is "deep" representation needed?

### Evaluation Metrics

- **Accuracy** (multiclass authorship), **AUROC** + Accuracy (binary modes).
- **Layer × N heatmap** (Sarfati Fig. 2B style).
- **Confusion matrix** for authorship (intra- vs. extra-author).
- **PCA explained-variance curves** for intrinsic dimensionality.
- **Cross-mode transfer matrix** (4×4 accuracy table).

### Statistical Analysis Plan

- 5-fold stratified CV; report mean ± std.
- Significance: compare probe accuracy against TF-IDF baseline with paired
  t-test across folds; α = 0.05.
- Effect sizes: Δaccuracy, Cohen's d on per-fold differences.

## Expected Outcomes

- **Supportive of hypothesis**: All four modes decodable >75%; clear monotone
  layer×N progression; PCA dim ≲20 for each; cross-mode cosine moderate but transfer
  weak.
- **Refuting**: AI-mode probe fails (would contradict published results — likely an
  implementation bug); register/typo modes near chance (would mean LLM doesn't
  encode them); cross-mode transfer ≈ within (would mean modes are not separable).

## Timeline and Milestones

| Phase | Time | Output |
|-------|------|--------|
| Setup (Llama download, env) | 15 min | Working model |
| Data prep (4 mode corpora) | 30 min | `data/chunks.parquet` |
| Embedding extraction (cached) | 30 min | `cache/embeddings.npz` |
| Probe training (all modes × layers × N) | 30 min | `results/probe_scores.csv` |
| Geometry analysis | 15 min | `results/geometry.csv` |
| Plots & report | 30 min | `figures/*.png`, REPORT.md |

Total budget: ~2.5h experiment + 30min documentation.

## Potential Challenges

- **Llama download size / time**: Mitigate by caching to HF dir; fall back to
  Pythia-1B if HF rate-limits.
- **HC3 length imbalance** (human answers shorter): chunk at fixed N to neutralize.
- **Synthetic register / typo data may be too easy**: include a TF-IDF baseline so we
  know whether probe is doing more than surface counting.
- **GPU OOM**: Llama-1B fits easily in a single A6000; batch size 32 safe.

## Success Criteria

- Pipeline runs end-to-end without errors.
- At least 3 of 4 modes show probe accuracy >>chance.
- Geometric analysis produces interpretable structure (Layer×N heatmap, cross-mode
  cosine matrix).
- REPORT.md states a clear answer to the research question grounded in our results.
