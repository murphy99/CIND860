"""
Cross-validation comparison of Logistic Regression, Naive Bayes, and Random
Forest on the IMDb sentiment dataset (TF-IDF features).

Reconstructed from a live Python session after the original notebook/script
that produced it was lost. Config (vectorizer params, model params, CV
scheme, scoring metrics) was read back out of the in-memory objects, so it
should exactly reproduce the original run.
"""

import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
train = pd.read_csv("data/IMDB Movies/imdb_train.csv")
test = pd.read_csv("data/IMDB Movies/imdb_test.csv")

X_text = train["text_clean"]
y = train["label"]

# ---------------------------------------------------------------------------
# CV scheme and scoring
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42
    ),
}

# ---------------------------------------------------------------------------
# Run cross-validation for each model (TF-IDF -> classifier pipeline)
# ---------------------------------------------------------------------------
results = {}

for name, model in models.items():
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_df=0.95, min_df=2, max_features=500),
            ),
            ("clf", model),
        ]
    )

    t0 = time.time()
    cv_results = cross_validate(
        pipe,
        X_text,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )
    elapsed = time.time() - t0

    results[name] = cv_results

    print(f"\n{name} (fit in {elapsed:.1f}s total across folds)")
    for metric in scoring:
        test_scores = cv_results[f"test_{metric}"]
        train_scores = cv_results[f"train_{metric}"]
        print(
            f"  {metric:10s}  test={test_scores.mean():.4f} (+/-{test_scores.std():.4f})"
            f"   train={train_scores.mean():.4f}"
        )

# ---------------------------------------------------------------------------
# results / cv_results now hold, per model:
#   fit_time, score_time,
#   test_accuracy, train_accuracy,
#   test_f1, train_f1,
#   test_roc_auc, train_roc_auc
# (5 values each, one per fold)
# ---------------------------------------------------------------------------
