# build_imdb_csv_binary.R
#
# Builds the cleaned, IMDb labeled-review CSV used throughout
# the CIND860 SHAP project, from the raw Stanford "aclImdb" review files.
#
# Input : data/IMDB Movies/aclImdb/{train,test}/{pos,neg}/*.txt
# Output: data/IMDB Movies/imdb_labeled_clean_binary.csv
#         data/IMDB Movies/IMDB_Binary_Class.csv
#


library(tidyverse)
library(fs)

data_dir <- "data/aclImdb"
output_csv <- "data/IMDB Movies/imdb_labeled_clean_binary.csv"
final_csv  <- "data/IMDB Movies/IMDB_Binary_Class.csv"

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

# --- 3. Drop exact train/test duplicates (cross-split only) ----------------
# Matches the canonical dedup approach in build_imdb_csv.R / build_imdb_csv_stars.R:
# if a cleaned review's text is byte-identical between the train and test
# splits, drop the train-side copy (test is left untouched).
cross_split_dupe_text <- intersect(
  labeled$text_clean[labeled$split == "train"],
  labeled$text_clean[labeled$split == "test"]
)
n_before <- nrow(labeled)
labeled <- labeled |>
  filter(!(split == "train" & text_clean %in% cross_split_dupe_text))
message(
  "Dropped ", n_before - nrow(labeled),
  " train-side rows that exactly duplicated a test-side review."
)

# --- 3b. Drop exact duplicates within each split (train and test) ----------
# After the cross-split dedup above, also remove exact text_clean duplicates
# occurring within the same split (keeps the first occurrence of each).
n_before_within <- nrow(labeled)
labeled <- labeled |>
  distinct(split, text_clean, .keep_all = TRUE)
message(
  "Dropped ", n_before_within - nrow(labeled),
  " additional within-split duplicate rows (train and test each deduplicated)."
)

# --- 4. Write the full cleaned, deduplicated labeled CSV --------------------
labeled |>
  select(text_raw, text_clean, label, split, rating) |>
  write_csv(output_csv)

message("Wrote ", nrow(labeled), " rows to ", output_csv)

# --- 5. Write single combined CSV with numeric labels -----------------------
imdb_export <- labeled |>
  mutate(label_num = if_else(label == "pos", 1L, 0L)) |>
  select(text_raw, text_clean, label = label_num)

write_csv(imdb_export, final_csv)

message("Wrote ", nrow(imdb_export), " rows to ", final_csv)