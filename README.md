# CIND860

## Dataset

This project uses the Stanford Large Movie Review Dataset ("aclImdb").

- **Download URL**:
 https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
  (official Stanford AI Lab mirror; ~80MB)

Update the ai.stanford.edu has been down, here is another mirror:

https://huggingface.co/datasets/NolanChai/aclImdb_v1/blob/main/aclImdb_v1.tar.gz


The archive is not committed to git (excluded via `.gitignore`, `data/*`), so after cloning this repo, download it into
`data/IMDB Movies/aclImdb_v1.tar.gz`, extract it, and then run
`scripts/build_imdb_csv.R` to regenerate the cleaned CSVs used for modeling.

## File structure (git-tracked files)

```
CIND860/
├── README.md
├── .gitignore
├── references.bib                     # Bibliography file
├── Abstract.qmd                       # Project abstract (source)
├── Abstract.pdf                       # Project abstract (rendered)
├── LitReview.qmd                      # Literature review (source)
├── LitReview.pdf                      # Literature review (rendered)
├── IMDb_EDA_Report.pdf                # EDA report (rendered)
├── Initial_Results.qmd                # Baseline modeling results (source - obsolete)
├── Initial_Results.ipynb              # Baseline modeling results (rendered notebook - obsolete)
├── Initial_Results_Binary.qmd         # Binary-class modeling results (source)
├── Initial_Results_multi.qmd          # Multi-class modeling results (source)
├── docs/
│   ├── Initial_Results_Binary.html    # Binary-class results (rendered)
│   ├── Initial_Results_Binary.pdf     # Binary-class results (rendered)
│   ├── Initial_Results_multi.html     # Multi-class results (rendered)
│   └── Initial_Results_multi.pdf      # Multi-class results (rendered)
└── scripts/
    ├── README.md
    ├── build_imdb_csv.R               # Builds imdb_labeled_clean/train/test.csv from raw aclImdb data
    ├── build_imdb_csv_binary.R        # Builds binary-class CSV (IMDB_Binary_Class.csv)
    ├── build_imdb_csv_stars.R         # Builds multi-class star-level CSV (IMDB_Stars_Class.csv)
    ├── cv_model_comparison.py         # Cross-validation comparison (LR/NB/RF)
    ├── pre_run_lstm_ver4.py           # Pre-processing for LSTM (attention variant)
    ├── run_lstm.py                    # Builds LSTM model in Pyenv tf311
    └── run_lstm_ver4.py               # LSTM with attention layer
```

