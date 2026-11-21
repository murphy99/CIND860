# build_imdb_csv_stars.R
#
# Builds the cleaned, IMDb labeled-review CSV used throughout
# the CIND860 SHAP project, from the raw Stanford "aclImdb" review files,
# adding a derived "star" field (1/2/4/5) on top of the binary pos/neg label.
#
# Star mapping (ratings 5-6 are excluded from the dataset by design):
#   rating 1-2  -> star 1
#   rating 3-4  -> star 2
#   rating 7-8  -> star 4
#   rating 9-10 -> star 5
#
# Input : data/aclImdb/{train,test}/{pos,neg}/*.txt
# Output: data/IMDB Movies/imdb_labeled_clean_stars.csv
#         data/IMDB Movies/IMDB_Stars_Class.csv
#
# Modeled on scripts/build_imdb_csv_binary.R.


library(tidyverse)
library(fs)

data_dir   <- "data/aclImdb"
output_csv <- "data/IMDB Movies/imdb_labeled_clean_stars.csv"
final_csv  <- "data/IMDB Movies/IMDB_Stars_Class.csv"

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
    str_replace_all("<br\\s*/?>", " ") |> #strip html
    str_replace_all("<[^>]+>", " ") |> #strip html
    str_to_lower() |> #lower case
    str_replace_all("[^a-z0-9'\\s]", " ") |>
    str_replace_all("\\s+", " ") |>          #White spaces
    str_replace_all("\\d+", "") %>%           # Remove numbers
    str_replace_all("[^a-zA-Z\\s]", "") %>%  # Keep only letters and spaces
    str_replace_all("\\s+", " ") %>%          # Replace multiple spaces with single space
    str_trim()
}

# --- 3. Derive the star field from rating -----------------------------------
rating_to_star <- function(rating) {
  case_when(
    rating %in% 1:2   ~ 1L,
    rating %in% 3:4   ~ 2L,
    rating %in% 7:8   ~ 4L,
    rating %in% 9:10  ~ 5L,
    TRUE              ~ NA_integer_
  )
}

labeled <- meta |>
  mutate(text_raw = map_chr(path, read_file)) |>
  mutate(
    text_clean = clean_review(text_raw),
    n_words    = str_count(text_clean, "\\S+"),
    star       = rating_to_star(rating)
  )

# --- 4. Drop exact train/test duplicates (cross-split only) ----------------
# Matches the canonical dedup approach in build_imdb_csv.R: if a cleaned
# review's text is byte-identical between the train and test splits, drop
# the train-side copy (test is left untouched). This does NOT remove
# duplicates that occur within the same split.
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

# --- 4b. Drop exact duplicates within each split (train and test) ----------
# After the cross-split dedup above, also remove exact text_clean duplicates
# occurring within the same split (keeps the first occurrence of each).
n_before_within <- nrow(labeled)
labeled <- labeled |>
  distinct(split, text_clean, .keep_all = TRUE)
message(
  "Dropped ", n_before_within - nrow(labeled),
  " additional within-split duplicate rows (train and test each deduplicated)."
)

# --- 5. Write the full cleaned labeled CSV (with star) ----------------------
labeled |>
  select(text_raw, text_clean, label, split, rating, star) |>
  write_csv(output_csv)

message("Wrote ", nrow(labeled), " rows to ", output_csv)

# --- 6. Write single combined CSV with numeric labels + star ----------------
imdb_export <- labeled |>
  mutate(label_num = if_else(label == "pos", 1L, 0L)) |>
  select(text_raw, text_clean, star)

write_csv(imdb_export, final_csv)

message("Wrote ", nrow(imdb_export), " rows to ", final_csv)

