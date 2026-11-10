# CIND860

## Dataset

This project uses the Stanford Large Movie Review Dataset ("aclImdb").

- **Download URL**:
 https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
  (official Stanford AI Lab mirror; ~80MB)

The archive is not committed to git (excluded via `.gitignore`, `data/*`), so after cloning this repo, download it into
`data/IMDB Movies/aclImdb_v1.tar.gz`, extract it, and then run
`scripts/build_imdb_csv.R` to regenerate the cleaned CSVs used for modeling.

## File structure (git-tracked files)

```
CIND860/
├── README.md
├── references.bib                # Bibliography file
├── Abstract.qmd                  # Project abstract (source)
├── Abstract.pdf                  # Project abstract (rendered)
├── LitReview.qmd                 # Literature review (source)
├── LitReview.pdf                 # Literature review (rendered)
├── Initial_Results.qmd           # Baseline modeling results (source)
├── Initial_Results.pdf           # Baseline modeling results (rendered)
└── scripts/
    ├── README.md
    ├── build_imdb_csv.R          # Builds imdb_labeled_clean/train/test.csv from raw aclImdb data
    └── run_lstm.py               # Builds lstm model in Pyenv tf311
    └── run_lstm_ver4.py          # Includes attention layer
```

