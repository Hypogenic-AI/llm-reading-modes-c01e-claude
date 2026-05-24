# Datasets for "Modes" in LLM Reading

Data files are NOT committed to git due to size. Follow the download instructions
below to reproduce.

## Dataset 1: Project Gutenberg Literary Corpus

### Overview
- **Source**: gutenberg.org (public domain)
- **Size**: 11 novels, ~7 MB total plain text
- **Format**: UTF-8 plain text with Gutenberg license headers
- **Task**: Author/style classification (replicating Sarfati et al. 2025 setup)
- **License**: Public domain (US works pre-1928)

### Contents
Authors and works in `datasets/gutenberg/`:
| Label | File | Author / Work |
|-------|------|---------------|
| GE | GE_Silas_Marner.txt | George Eliot — Silas Marner |
| JA1 | JA1_Pride_and_Prejudice.txt | Jane Austen — Pride and Prejudice |
| JA2 | JA2_Sense_and_Sensibility.txt | Jane Austen — Sense and Sensibility |
| JA3 | JA3_Emma.txt | Jane Austen — Emma |
| HM1 | HM1_Moby_Dick.txt | Herman Melville — Moby-Dick |
| HM2 | HM2_Typee.txt | Herman Melville — Typee |
| MT1 | MT1_Tom_Sawyer.txt | Mark Twain — Tom Sawyer |
| MT2 | MT2_Huckleberry_Finn.txt | Mark Twain — Huckleberry Finn |
| NH1 | NH1_Scarlet_Letter.txt | Nathaniel Hawthorne — The Scarlet Letter |
| JJ | JJ_Ulysses.txt | James Joyce — Ulysses |
| GS | GS_Sherlock.txt | Arthur Conan Doyle — Adventures of Sherlock Holmes |

### Download
```bash
python -c "
import requests, os
os.makedirs('datasets/gutenberg', exist_ok=True)
urls = {
    'GE_Silas_Marner.txt':           'https://www.gutenberg.org/files/550/550-0.txt',
    'JA1_Pride_and_Prejudice.txt':   'https://www.gutenberg.org/files/1342/1342-0.txt',
    'JA2_Sense_and_Sensibility.txt': 'https://www.gutenberg.org/files/161/161-0.txt',
    'JA3_Emma.txt':                  'https://www.gutenberg.org/files/158/158-0.txt',
    'HM1_Moby_Dick.txt':             'https://www.gutenberg.org/files/2701/2701-0.txt',
    'HM2_Typee.txt':                 'https://www.gutenberg.org/files/1900/1900-0.txt',
    'MT1_Tom_Sawyer.txt':            'https://www.gutenberg.org/files/74/74-0.txt',
    'MT2_Huckleberry_Finn.txt':      'https://www.gutenberg.org/files/76/76-0.txt',
    'NH1_Scarlet_Letter.txt':        'https://www.gutenberg.org/files/25344/25344-0.txt',
    'JJ_Ulysses.txt':                'https://www.gutenberg.org/files/4300/4300-0.txt',
    'GS_Sherlock.txt':               'https://www.gutenberg.org/files/1661/1661-0.txt',
}
for name, url in urls.items():
    r = requests.get(url, headers={'User-Agent':'Mozilla/5.0'})
    open(f'datasets/gutenberg/{name}', 'wb').write(r.content)
"
```

### Loading
```python
text = open('datasets/gutenberg/GE_Silas_Marner.txt', encoding='utf-8').read()
# Strip Gutenberg header/footer between '*** START OF' and '*** END OF' markers.
```

---

## Dataset 2: HC3 — Human ChatGPT Comparison Corpus

### Overview
- **Source**: Hello-SimpleAI on HuggingFace (https://huggingface.co/datasets/Hello-SimpleAI/HC3)
- **Paper**: Guo et al. 2023, "How Close is ChatGPT to Human Experts?" (arXiv:2301.07597)
- **Size**: ~24K questions × {human_answers, chatgpt_answers}, ~71 MB English
- **Format**: JSONL, one record per question
- **Task**: Human vs. ChatGPT binary detection; mode probing
- **License**: CC-BY-SA (with deference to source licenses)

### Schema
Each line of `all.jsonl`:
```json
{
  "question": "...",
  "human_answers": ["answer1", "answer2", ...],
  "chatgpt_answers": ["answer1", ...],
  "index": null,
  "source": "reddit_eli5 | finance | medicine | open_qa | wiki_csai"
}
```

### Splits by source domain
- `reddit_eli5.jsonl` — ELI5 from Reddit (~54 MB)
- `finance.jsonl` — Finance Q&A (~10 MB)
- `medicine.jsonl` — Medical Q&A (~3 MB)
- `open_qa.jsonl` — Open-domain Q&A (~3 MB)
- `wiki_csai.jsonl` — Wikipedia CS/AI (~2 MB)

### Download
```bash
python -c "
import requests, os
os.makedirs('datasets/HC3', exist_ok=True)
files = ['all.jsonl', 'reddit_eli5.jsonl', 'finance.jsonl', 'medicine.jsonl', 'open_qa.jsonl', 'wiki_csai.jsonl']
for f in files:
    url = f'https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/{f}'
    r = requests.get(url, timeout=60)
    open(f'datasets/HC3/{f}', 'wb').write(r.content)
"
```

### Loading
```python
import json
with open('datasets/HC3/all.jsonl') as f:
    rows = [json.loads(l) for l in f]

# Construct binary dataset
human, ai = [], []
for r in rows:
    for a in (r.get('human_answers') or []):
        if a and a.strip(): human.append(a)
    for a in (r.get('chatgpt_answers') or []):
        if a and a.strip(): ai.append(a)
```

See `datasets/HC3/samples.json` for 10 reference records.

---

## Dataset 3 (Optional, larger): M4 — Multi-generator MGT Detection

### Overview
Comes bundled inside `code/M4/data/` (~416 MB). Skip download — already cloned.

- **Source**: https://github.com/mbzuai-nlp/M4
- **Domains**: Wikipedia, WikiHow, Reddit ELI5, arXiv, PeerRead (English); plus multilingual
- **Generators**: GPT-4, ChatGPT, GPT-3.5 (davinci), Cohere, Dolly-v2, BLOOMz, FlanT5, LLaMA
- **Format**: JSONL files, one per (domain, generator) pair
- **Task**: Multi-class generator attribution + binary human/AI detection
- **Citation**: Wang et al. 2024, EACL — arXiv:2305.14902

### Loading
```python
import json
with open('code/M4/data/arxiv_chatGPT.jsonl') as f:
    rows = [json.loads(l) for l in f]
```

---

## Dataset 4 (Optional): Comprehensive Human-vs-AI Dataset

- **Source**: https://huggingface.co/datasets/gsingh1-py/train
- **Paper**: Roy et al. 2025 (`papers/2025_Roy_ComprehensiveDataset.pdf`)
- **Size**: 7,321 rows / 161 MB
- **Models**: Gemma-2-9B, Mistral-7B, Qwen-2-72B, LLaMA-8B, Yi-Large, GPT-4o
- **Columns**: `prompt`, `Human_story`, plus one column per model

### Download
```python
from datasets import load_dataset
ds = load_dataset('gsingh1-py/train')
ds.save_to_disk('datasets/roy_comprehensive')
```

---

## What's NOT here (but is in the literature)

- **Dictated speech transcripts (ASR-style)**: No off-the-shelf dataset of paired
  "dictated text vs. written text" with the same content was found. Candidate workaround:
  use LibriSpeech transcripts (read aloud) vs. matched written paragraphs, or generate
  synthetic dictation via TTS→ASR pipelines.
- **Keyboard-error / typing-mistake corpora**: No widely used "user-typed-on-QWERTY"
  vs. "user-typed-on-Dvorak" labeled dataset. Workaround: synthetically inject typos
  with a known keyboard-adjacency model.
- **Influence-by-typing-keyboard**: This is the most speculative arm of the hypothesis.
  Likely needs synthetic data generation as a first experiment.

These gaps are notable for experiment design: the easiest "modes" to study empirically
are (1) human vs. LLM and (2) author/style. Other modes (dictation, keyboard layout)
likely require synthetic data creation as part of the experimental work.
