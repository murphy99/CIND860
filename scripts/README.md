# scripts/

## build_imdb_csv.R

Builds the cleaned, deduplicated IMDb labeled-review CSVs used throughout the
CIND860 SHAP project.

**Input**: `data/IMDB Movies/aclImdb/{train,test}/{pos,neg}/*.txt`
(extracted from `aclImdb_v1.tar.gz`)

**Output**:
- `data/IMDB Movies/imdb_labeled_clean.csv`
  (50,000 rows: `text_raw`, `text_clean`, `label`, `split`, `rating`)
- `data/IMDB Movies/imdb_train.csv` (25,000 rows) and
  `data/IMDB Movies/imdb_test.csv` (25,000 rows) — Python-ready train/test
  splits derived from the file above (`text_raw`, `text_clean`, `label` as
  numeric 0/1, `rating`; no `split` column).

### What it does

1. Builds file metadata (split/label/id/rating) for all labeled train+test
   reviews (`unsup` reviews are excluded).
2. Reads raw review text and cleans it: strips `<br />`/HTML tags, lowercases,
   removes punctuation except apostrophes, collapses whitespace.
4. Writes the final pre-processed into  `imdb_labeled_clean.csv`.
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

Verified 2026-10-19: running the script standalone (fresh `Rscript` process,
no inherited session state, output redirected to a scratch directory)
reproduces the existing on-disk CSVs byte-for-byte in content: 49,877 rows
for `imdb_labeled_clean.csv` (123 exact train-side duplicates dropped),
24,877 rows for `imdb_train.csv`, 25,000 rows for `imdb_test.csv`. Row order
can differ between runs because `fs::dir_ls()` relies on filesystem
directory-listing order, which is not guaranteed to be stable — this has no
effect on content, row count, or class balance.
