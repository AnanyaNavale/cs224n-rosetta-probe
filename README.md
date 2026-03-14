# The Rosetta Probe: Cross-Lingual Syntactic Transfer in Monolingual English BERT

CS224N Custom Project — Stanford University
Author: Ananya Navale

## Overview
This project investigates whether monolingual English BERT embeddings encode recoverable 
syntactic information across seven non-English languages spanning Germanic, Romance, Slavic, 
and Uralic families, using the structural probe framework of Hewitt & Manning (2019).

## Repository Structure
```
├── notebook.ipynb          # Main Colab notebook with full pipeline
├── scripts/
│   ├── convert_conll_to_raw.py      # Converts CoNLL-U files to raw text
│   ├── convert_raw_to_bert.py       # Extracts BERT embeddings to HDF5
│   ├── compute_linear_baseline.py   # Computes LINEAR baseline (original contribution)
│   ├── reporter.py                  # Modified: UPOS punctuation filtering fix
│   ├── data.py                      # Modified: CoNLL-U multi-word token fix
│   └── run_experiment.py            # Modified: yaml.load fix
├── config/                 # YAML experiment configs for all 8 languages
├── visualizations/         # Generated bar charts and figures
└── results/                # Probe outputs (tikz, PNG, scores) for all languages
```

## Pipeline
For each language, the pipeline follows four steps:

1. **Data preparation** — clone UD treebank, truncate to 11K sentences, remove sentences >512 tokens
2. **BERT extraction** — run `convert_raw_to_bert.py` to generate HDF5 embeddings
3. **Probe training & evaluation** — run `run_experiment.py` with the appropriate config file
4. **LINEAR baseline** — run `compute_linear_baseline.py` on test CoNLL-U files

## Languages & Treebanks
| Language | Treebank | Family |
|----------|----------|--------|
| English | UD_English-EWT | Germanic |
| German | UD_German-GSD | Germanic |
| Dutch | UD_Dutch-Alpino | Germanic |
| French | UD_French-GSD | Romance |
| Italian | UD_Italian-ISDT | Romance |
| Spanish | UD_Spanish-AnCora | Romance |
| Czech | UD_Czech-PDTC | Slavic |
| Finnish | UD_Finnish-TDT | Uralic |

## Original Contributions
- `compute_linear_baseline.py` — novel script computing LINEAR baseline across all languages
- `reporter.py` — fixed punctuation filtering to use universal UPOS PUNCT tag instead of 
  language-specific XPOS tags, enabling cross-lingual evaluation
- `data.py` and `convert_conll_to_raw.py` — fixed CoNLL-U multi-word token handling

## Dependencies
- Python 3.12
- PyTorch
- pytorch-pretrained-bert
- h5py
- scipy
- numpy
- matplotlib

## Base Codebase
This project builds on the structural probe codebase by Hewitt & Manning (2019):
https://github.com/john-hewitt/structural-probes

## Reference
Hewitt, J., & Manning, C. D. (2019). A structural probe for finding syntax in word 
representations. NAACL 2019.
