# scripts/

## build_imdb_csv.R

Builds the cleaned, deduplicated IMDb labeled-review CSV used throughout the
CIND860 SHAP project.

**Input**: `data/IMDB Movies/aclImdb/{train,test}/{pos,neg}/*.txt`
(extracted from `aclImdb_v1.tar.gz`)

**Output**: `data/IMDB Movies/imdb_labeled_clean.csv`
(49,874 rows: `text_raw`, `text_clean`, `label`, `split`, `rating`)

### What it does

1. Builds file metadata (split/label/id/rating) for all labeled train+test
   reviews (`unsup` reviews are excluded).
2. Reads raw review text and cleans it: strips `<br />`/HTML tags, lowercases,
   removes punctuation except apostrophes, collapses whitespace.
3. Removes exact cross-split duplicate reviews (drops the train-side copy).
4. Removes near-duplicate cross-split reviews — reviews that open with the
   same 8 words and have 3-gram shingle Jaccard similarity > 0.5 — again
   dropping the train-side copy.
5. Writes the final deduplicated set to CSV.

### When to re-run it

- You don't need to re-run this for normal analysis — just load the existing
  `data/IMDB Movies/imdb_labeled_clean.csv` directly.
- Re-run it only if:
  - the raw `aclImdb` data changes or is re-extracted from the archive, or
  - the CSV is lost/corrupted and needs to be regenerated from source, or
  - you intentionally want to change the cleaning or dedup logic (edit the
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
(reads and cleans 50,000 review files, then does pairwise near-duplicate
comparison within blocked groups).

### Verification

Verified 2026-10-16: running the script standalone (fresh `Rscript` process,
no inherited session state) reproduces the existing
`imdb_labeled_clean.csv` byte-for-byte in content (49,874 rows; 123 exact
train-side duplicates dropped; 4 near-duplicate candidate pairs found, max
similarity 0.798, 3 unique train-side rows dropped). Row order can differ
between runs because `fs::dir_ls()` relies on filesystem directory-listing
order, which is not guaranteed to be stable — this has no effect on content,
row count, or class balance.
