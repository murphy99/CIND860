"""
run_lstm.py

Trains and evaluates an LSTM (RNN) sentiment classifier on imdb_train.csv,
using the same 80/20 stratified holdout split established in
Initial_Results.qmd (test_size=0.2, stratify=y, random_state=42) -- no
cross-validation, per project decision (LSTM training is too expensive to
repeat 5x).

v2 (2026-11-02): first attempt (random embedding init, LSTM(64) +
Dense(32) + Dropout, lr=1e-3) scored only ~0.52 test accuracy -- barely
above chance, with validation loss rising almost immediately (overfitting
onset before the model learned much signal). This version addresses that
with:
  - Pre-trained GloVe embeddings (glove.6B.100d, frozen / non-trainable)
    instead of randomly-initialized embeddings learned from scratch on
    only 20,000 reviews.
  - Lower learning rate (1e-4 instead of 1e-3).
  - Simpler architecture: single LSTM(32) straight to Dense(1, sigmoid),
    no extra Dense(32)+Dropout stack, no recurrent_dropout.
  Result: accuracy 0.6508, f1 0.6504, roc_auc 0.6681 -- big improvement
  over v1, but still well short of the TF-IDF + Logistic Regression
  baseline (accuracy 0.886).

v3 (2026-11-02): same frozen GloVe embeddings and lr=1e-4 as v2, but
swaps the unidirectional LSTM(32) for a Bidirectional LSTM(64) (so 64
units per direction, 128 total) to add capacity, per project decision
after reviewing v2's results.

Must be run with the tf311 environment's interpreter (TensorFlow 2.16.2,
Python 3.11.9), e.g.:

    ~/tf311/bin/python scripts/run_lstm.py

NOT the cind860 environment -- TensorFlow requires numpy<2, which conflicts
with cind860's shap/scipy stack (numpy>=2). See AGENTS.md for details.

GloVe embeddings: data/glove/glove.6B.100d.txt (400,000 words, 100-dim),
downloaded from a HuggingFace mirror of Stanford's GloVe vectors (Stanford's
own nlp.stanford.edu refuses direct connections from this network).
Not tracked in git (see .gitignore) -- re-download if missing.
"""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

DATA_DIR = "data/IMDB Movies"
GLOVE_PATH = "data/glove/glove.6B.100d.txt"
MAX_WORDS = 10000      # vocabulary size, matches TF-IDF baseline's max_features
MAX_LEN = 450           # covers ~90% of reviews per project EDA findings
EMBED_DIM = 100         # must match the GloVe file chosen (100d)
LSTM_UNITS = 64         # per direction; Bidirectional doubles this to 128 total
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-4    # lower than v1's 1e-3
RANDOM_STATE = 42

print("Loading data...")
train = pd.read_csv(f"{DATA_DIR}/imdb_train.csv")
print("train shape:", train.shape)

X_text = train["text_clean"]
y = train["label"]

# Same 80/20 stratified split as Initial_Results.qmd / v1 of this script
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("X_train:", X_train.shape, " X_test:", X_test.shape)

print("\nTokenizing...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

print("X_train_pad:", X_train_pad.shape, " X_test_pad:", X_test_pad.shape)

print("\nLoading GloVe embeddings...")
t0 = time.time()
embeddings_index = {}
with open(GLOVE_PATH, encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
print(f"Loaded {len(embeddings_index)} word vectors ({time.time() - t0:.1f}s)")

word_index = tokenizer.word_index  # word -> rank (1-indexed), capped by num_words at lookup time
num_vocab = min(MAX_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_vocab, EMBED_DIM))
hits, misses = 0, 0
for word, i in word_index.items():
    if i >= num_vocab:
        continue
    vec = embeddings_index.get(word)
    if vec is not None:
        embedding_matrix[i] = vec
        hits += 1
    else:
        misses += 1
print(f"Embedding matrix: {embedding_matrix.shape}, {hits} hits / {misses} misses "
      f"({hits / (hits + misses) * 100:.1f}% coverage)")

print("\nBuilding model...")
model = Sequential([
    Embedding(
        input_dim=num_vocab,
        output_dim=EMBED_DIM,
        weights=[embedding_matrix],
        trainable=False,  # frozen pre-trained embeddings
    ),
    Bidirectional(LSTM(LSTM_UNITS)),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.build(input_shape=(None, MAX_LEN))
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

print("\nTraining...")
t0 = time.time()
history = model.fit(
    X_train_pad, y_train,
    validation_split=0.1,  # carve a small slice of X_train for early stopping only
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=2,
)
elapsed = time.time() - t0
print(f"\nTraining elapsed: {elapsed:.1f}s ({len(history.history['loss'])} epochs run)")

print("\nEvaluating on held-out X_test (20% split, untouched during training)...")
y_pred_proba = model.predict(X_test_pad, batch_size=BATCH_SIZE).ravel()
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nLSTM (GloVe, v3, BiLSTM) test results:")
print(f"  accuracy: {acc:.4f}")
print(f"  f1:       {f1:.4f}")
print(f"  roc_auc:  {auc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["neg", "pos"]))
