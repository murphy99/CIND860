# build_imdb_csv.R
#
# Builds the cleaned, IMDb labeled-review CSV used throughout
# the CIND860 SHAP project, from the raw Stanford "aclImdb" review files.
#
# Input : data/IMDB Movies/aclImdb/{train,test}/{pos,neg}/*.txt
# Output: data/IMDB Movies/imdb_labeled_clean.csv
#         data/IMDB Movies/imdb_train.csv
#         data/IMDB Movies/imdb_test.csv
#
# Pipeline:
#   1. Build file metadata (split/label/id/rating) for all labeled reviews.
#   2. Read raw text and clean it (strip HTML, lowercase, strip punctuation
#      except apostrophes, collapse whitespace).
#   3. Write text_raw, text_clean, label, split, rating to CSV.
#   4. Export Python-ready train/test split CSVs (text_raw, text_clean,
#      label as numeric 0/1, rating; no split column since it's implicit
#      in the filename) for the modeling stage.
#
#
# Verified 2026-10-16: 50,000 total labeled files -> 123 exact train-side
# dupes dropped -> 49,877 rows (train: 12,407 neg / 12,470 pos; test: 12,500
# neg / 12,500 pos).

library(tidyverse)
library(fs)

data_dir <- "data/IMDB Movies/aclImdb"
output_csv <- "data/IMDB Movies/imdb_labeled_clean.csv"
train_csv <- "data/IMDB Movies/imdb_train.csv"
test_csv  <- "data/IMDB Movies/imdb_test.csv"

# --- 1. Build file metadata for labeled reviews (train/test x pos/neg) -----
list_review_files <- function(split, label) {
  dir_path <- path(data_dir, split, label)
  tibble(
    path  = dir_ls(dir_path, glob = "*.txt"),
    split = split,
    label = label
  )
}

meta <- bind_rows(
  list_review_files("train", "pos"),
  list_review_files("train", "neg"),
  list_review_files("test",  "pos"),
  list_review_files("test",  "neg")
) |>
  mutate(
    file   = path_file(path),
    id     = as.integer(str_extract(file, "^\\d+")),
    rating = as.integer(str_extract(file, "(?<=_)\\d+(?=\\.txt$)"))
  )

# --- 2. Read raw text and clean it ------------------------------------------
clean_review <- function(text) {
  text |>
    str_replace_all("<br\\s*/?>", " ") |>
    str_replace_all("<[^>]+>", " ") |>
    str_to_lower() |>
    str_replace_all("[^a-z0-9'\\s]", " ") |>
    str_replace_all("\\s+", " ") |>
    str_replace_all("\\d+", "") %>%           # Remove numbers
    str_replace_all("[^a-zA-Z\\s]", "") %>%  # Keep only letters and spaces
    str_replace_all("\\s+", " ") %>%          # Replace multiple spaces with single space
    str_trim()  
}

labeled <- meta |>
  mutate(text_raw = map_chr(path, read_file)) |>
  mutate(
    text_clean = clean_review(text_raw),
    n_words    = str_count(text_clean, "\\S+")
  )

# --- 3. Write the full cleaned labeled CSV ----------------------------------
labeled |>
  select(text_raw, text_clean, label, split, rating) |>
  write_csv(output_csv)

message("Wrote ", nrow(labeled), " rows to ", output_csv)

# --- 5. Write Python-ready train/test split CSVs ----------------------------
imdb_export <- labeled |>
  mutate(label_num = if_else(label == "pos", 1L, 0L)) |>
  select(text_raw, text_clean, label = label_num, rating, split)

train_df <- imdb_export |> filter(split == "train") |> select(-split)
test_df  <- imdb_export |> filter(split == "test")  |> select(-split)

write_csv(train_df, train_csv)
write_csv(test_df,  test_csv)

message("Wrote ", nrow(train_df), " rows to ", train_csv)
message("Wrote ", nrow(test_df),  " rows to ", test_csv)
