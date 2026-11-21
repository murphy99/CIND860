# scripts/

## build_imdb_csv.R

Builds the cleaned, deduplicated IMDb labeled-review CSVs used throughout the
CIND860 SHAP project.

**Input**: `data/aclImdb/{train,test}/{pos,neg}/*.txt`
(extracted from `aclImdb_v1.tar.gz`)

**Output**:
- `data/IMDB Movies/imdb_labeled_clean.csv`
  (`text_raw`, `text_clean`, `label`, `split`, `rating`)
- `data/IMDB Movies/imdb_train.csv` and
  `data/IMDB Movies/imdb_test.csv` — Python-ready train/test
  splits derived from the file above (`text_raw`, `text_clean`, `label` as
  numeric 0/1, `rating`; no `split` column).

### What it does

1. Builds file metadata (split/label/id/rating) for all labeled train+test
   reviews (`unsup` reviews are excluded).
2. Reads raw review text and cleans it: strips `<br />`/HTML tags, lowercases,
   removes punctuation except apostrophes, collapses whitespace.
3. Drops exact cross-split duplicates (a train-side review that's
   byte-identical to a test-side review is dropped from train), then drops
   exact duplicates occurring within the same split (train and test each
   deduplicated independently, keeping the first occurrence).
4. Writes the final pre-processed data to `imdb_labeled_clean.csv`.
5. Exports the `imdb_train.csv`/`imdb_test.csv` split CSVs for the modeling
   stage.

### When to re-run it

- You don't need to re-run this for normal analysis — just load the existing
  `data/IMDB Movies/imdb_labeled_clean.csv` directly.
- Re-run it only if:
  - the raw `aclImdb` data changes or is re-extracted from the archive, or
  - the CSV is lost/corrupted and needs to be regenerated from source, or
  - you intentionally want to change the cleaning logic (edit the
    script, then re-run to regenerate the CSV).

### How to run it

From the project root (`CIND860/`), in a fresh R session:

```r
source("scripts/build_imdb_csv.R")
```

or from the shell:

```sh
Rscript scripts/build_imdb_csv.R
```

Requires the `tidyverse` and `fs` packages. Takes roughly one minute to run
(reads and cleans 50,000 review files).

### Verification

Verified 2026-11-16 (after within-split dedup was added on top of the
original cross-split dedup): 123 train-side rows dropped as exact
cross-split duplicates, plus 301 further rows dropped as within-split
duplicates, giving `imdb_labeled_clean.csv` = 49,576 rows,
`imdb_train.csv` = 24,778 rows, `imdb_test.csv` = 24,798 rows (train:
12,336 neg / 12,442 pos; test: 12,359 neg / 12,439 pos). Row order can
differ between runs because `fs::dir_ls()` relies on filesystem
directory-listing order, which is not guaranteed to be stable — this has no
effect on content, row count, or class balance.

## build_imdb_csv_binary.R

Same pipeline as `build_imdb_csv.R` (same input, same cleaning and two-stage
cross-split + within-split dedup), but writes to a separate pair of output
files intended for the binary-classification modeling notebooks.

**Input**: `data/aclImdb/{train,test}/{pos,neg}/*.txt`

**Output**:
- `data/IMDB Movies/imdb_labeled_clean_binary.csv` (`text_raw`, `text_clean`,
  `label`, `split`, `rating`)
- `data/IMDB Movies/IMDB_Binary_Class.csv` — combined (train+test not split),
  numeric `label` (0/1), `text_raw`, `text_clean` only

### How to run it

```r
source("scripts/build_imdb_csv_binary.R")
```

## build_imdb_csv_stars.R

Same pipeline again, but additionally derives a multi-class `star` field
(1/2/4/5) from the 1-10 `rating`, for finer-grained analysis than plain
pos/neg (ratings 5-6 are excluded from the dataset by design, so there is no
3-star bucket):

- rating 1-2 → star 1
- rating 3-4 → star 2
- rating 7-8 → star 4
- rating 9-10 → star 5

**Input**: `data/aclImdb/{train,test}/{pos,neg}/*.txt`

**Output**:
- `data/IMDB Movies/imdb_labeled_clean_stars.csv` (`text_raw`, `text_clean`,
  `label`, `split`, `rating`, `star`)
- `data/IMDB Movies/IMDB_Stars_Class.csv` — combined (train+test not split),
  `text_raw`, `text_clean`, `star`

### How to run it

```r
source("scripts/build_imdb_csv_stars.R")
```

All three build scripts (`build_imdb_csv.R`, `build_imdb_csv_binary.R`,
`build_imdb_csv_stars.R`) use the identical cross-split-then-within-split
dedup logic and agree exactly on counts (123 cross-split + 301 within-split
rows dropped, 49,576 rows in every non-split-combined output).

## cv_model_comparison.py

Cross-validation comparison of Logistic Regression, Naive Bayes, and Random
Forest on the IMDb sentiment dataset, using TF-IDF features in a
`Pipeline`. Reconstructed from a live Python session after the original
notebook/script that produced it was lost — config (vectorizer params,
model params, CV scheme, scoring metrics) was read back out of the
in-memory objects, so it should reproduce the original run.

**Input**: `data/IMDB Movies/imdb_train.csv`, `data/IMDB Movies/imdb_test.csv`

**What it does**: 5-fold `StratifiedKFold` (shuffled, `random_state=42`) via
`cross_validate(..., return_train_score=True)`, scored on accuracy, f1, and
roc_auc, for each of the three models in turn (TF-IDF → classifier
pipeline). Prints train/test mean ± std for each metric.

### How to run it

```sh
python scripts/cv_model_comparison.py
```

Requires `pandas` and `scikit-learn`.

## LSTM scripts (run_lstm.py, run_lstm_ver4.py, pre_run_lstm_ver4.py)

Train and evaluate an LSTM (RNN) sentiment classifier on `imdb_train.csv`,
using the same 80/20 stratified holdout split established in
`Initial_Results.qmd` (`test_size=0.2, stratify=y, random_state=42`) — no
cross-validation, since LSTM training is too expensive to repeat 5x.

**Must be run with the `tf311` environment's interpreter** (TensorFlow
2.16.2, Python 3.11.9), e.g.:

```sh
~/tf311/bin/python scripts/run_lstm.py
```

NOT the `cind860` environment — TensorFlow requires `numpy<2`, which
conflicts with `cind860`'s shap/scipy stack (`numpy>=2`).

- **`run_lstm.py`** (v3): uses frozen, pre-trained GloVe embeddings
  (`glove.6B.100d`, non-trainable) and a single Bidirectional LSTM(64) straight
  to `Dense(1, sigmoid)`, `lr=1e-4`. Result: accuracy 0.6508, f1 0.6504,
  roc_auc 0.6681 — a large improvement over an earlier random-embedding
  attempt, but still well short of the TF-IDF + Logistic Regression baseline
  (accuracy ~0.89).
- **`pre_run_lstm_ver4.py`**: standalone snippet that loads GloVe embeddings
  (`data/glove/glove.6B.100d.txt`) while filtering out NLTK English stop
  words, as a pre-processing step explored ahead of `run_lstm_ver4.py`. Not a
  complete, runnable pipeline on its own (data loading and the rest of the
  training code are stubbed out/commented); requires `nltk` with the
  `stopwords` corpus downloaded.
- **`run_lstm_ver4.py`** (v4): unfreezes the embeddings (`trainable=True`),
  increases LSTM capacity to 128 units per direction (256 total,
  Bidirectional), adds dropout and L2 regularization on the embeddings, an
  attention mechanism over the LSTM outputs, a lower learning rate (`5e-5`),
  and a larger validation split (20%). Also saves the trained model
  (`BiLSTM_imdb_sentiment_model.keras`), the tokenizer (`tokenizer.pkl`), a
  model architecture diagram (`model_architecture.png`), and per-review
  attention-weight analysis (`attention_analysis.csv`) alongside
  prediction/metric CSVs (`test_results.csv`, `test_metrics.csv`).

GloVe embeddings (`data/glove/glove.6B.100d.txt`, 400,000 words, 100-dim) are
downloaded from a HuggingFace mirror of Stanford's GloVe vectors (Stanford's
own `nlp.stanford.edu` refuses direct connections from this network). Not
tracked in git (see `.gitignore`) — re-download if missing.
