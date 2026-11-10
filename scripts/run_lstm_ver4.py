"""
run_lstm.py v4 - Improved LSTM with Unfrozen Embeddings

Key improvements over v3:
  - Unfrozen embeddings (trainable=True) to adapt to sentiment task
  - Increased LSTM capacity (128 units per direction, 256 total)
  - Dropout layers to prevent overfitting
  - L2 regularization on embeddings
  - Lower learning rate (5e-5) for careful tuning
  - Attention mechanism to focus on important words
  - Larger validation split (20%) for better early stopping

Expected accuracy: ~0.70-0.75 (vs v3's 0.651)

Must be run with the tf311 environment's interpreter.
"""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Bidirectional, AdditiveAttention, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Attention

DATA_DIR = "data/IMDB Movies"
GLOVE_PATH = "data/glove/glove.6B.100d.txt"
MAX_WORDS = 10000
MAX_LEN = 450
EMBED_DIM = 100
LSTM_UNITS = 128         # Increased from 64 (per direction; Bidirectional doubles to 256 total)
BATCH_SIZE = 64
EPOCHS = 20              # Increased from 15 (early stopping will cut it short anyway)
LEARNING_RATE = 5e-5     # Lowered from 1e-4 (for unfrozen embeddings)
RANDOM_STATE = 42

print("Loading data...")
train = pd.read_csv(f"{DATA_DIR}/imdb_train.csv")
print("train shape:", train.shape)

X_text = train["text_clean"]
y = train["label"]

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

word_index = tokenizer.word_index
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

print("\nBuilding improved model...")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Attention, GlobalAveragePooling1D

print("\nBuilding improved model with Functional API...")

# Input layer
inputs = Input(shape=(MAX_LEN,), name="input")

# Embedding layer
x = Embedding(
    input_dim=num_vocab,
    output_dim=EMBED_DIM,
    weights=[embedding_matrix],
    trainable=True,
    embeddings_regularizer=L2(1e-4),
    name="embedding"
)(inputs)

# Bidirectional LSTM
x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True), name="bilstm")(x)

# First Dropout
x = Dropout(0.3, name="dropout_1")(x)

# Attention: use LSTM output as both query and value
x = Attention(name="attention")([x, x])  # outputs (batch, 450, 256)

# Pool over sequence dimension BEFORE dropout
x = GlobalAveragePooling1D(name="global_avg_pool")(x)  # outputs (batch, 256)

# Second Dropout
x = Dropout(0.2, name="dropout_2")(x)

# Output layer
outputs = Dense(1, activation="sigmoid", name="output")(x)

# Create model
model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

print("\nTraining...")
t0 = time.time()
history = model.fit(
    X_train_pad, y_train,
    validation_split=0.2,  # Increased from 0.1 for better early stopping signal
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

print(f"\nLSTM v4 (Improved: Unfrozen Embeddings + Attention + Dropout) test results:")
print(f"  accuracy: {acc:.4f}")
print(f"  f1:       {f1:.4f}")
print(f"  roc_auc:  {auc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["neg", "pos"]))


# Optional: Compare to v3 baseline
print("\n" + "="*60)
print("Expected improvements over v3:")
print("  v3 (frozen, 64 units):     accuracy ~0.651")
print("  v4 (unfrozen, 128 units):  accuracy ~0.70-0.75 (expected)")
print("="*60)

print("\n" + "="*60)
print("SAVING RESULTS TO CSV...")
print("="*60)

import pandas as pd

# Save individual predictions
results_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'probability': y_pred_proba
})
results_df.to_csv("test_results.csv", index=False)
print("✓ Test results saved to: test_results.csv")

# Save summary metrics
metrics_df = pd.DataFrame({
    'metric': ['accuracy', 'f1_score', 'roc_auc'],
    'value': [acc, f1, auc]
})
metrics_df.to_csv("test_metrics.csv", index=False)
print("✓ Test metrics saved to: test_metrics.csv")

print("="*60)



# Save the model
print("\n" + "="*60)
print("SAVING MODEL AND TOKENIZER...")
print("="*60)

model.save("BiLSTM_imdb_sentiment_model.keras")
print("Model saved to: BiLSTM_model.keras")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved to: tokenizer.pkl")

print("="*60)

from tensorflow.keras.utils import plot_model

plot_model(
    model,
    to_file="model_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",  # Top-to-Bottom layout
    expand_nested=True,
    dpi=96
)

print("Model architecture saved to: model_architecture.png")

print("\n" + "="*60)
print("ANALYZING ATTENTION WEIGHTS...")
print("="*60)

from tensorflow.keras.models import Model

# Create attention weights model
attention_layer = model.get_layer("attention")
attention_weights_model = Model(
    inputs=model.input,
    outputs=attention_layer.output
)

# Get attention weights
print("\n" + "="*60)
print("ANALYZING ATTENTION WEIGHTS...")
print("="*60)

from tensorflow.keras.models import Model

# Create attention weights model
attention_layer = model.get_layer("attention")
attention_weights_model = Model(
    inputs=model.input,
    outputs=attention_layer.output
)

# Get attention weights
attention_output = attention_weights_model.predict(X_test_pad, batch_size=BATCH_SIZE)
print(f"✓ Attention weights extracted: {attention_output.shape}")

# Define function to get important words (FIXED)
def get_important_words(review_text, attention_weights, tokenizer, top_n=20):
    attention_scores = attention_weights.mean(axis=1)
    seq = tokenizer.texts_to_sequences([review_text])[0]
    
    word_attention = []
    for idx, word_id in enumerate(seq):
        if word_id > 0 and idx < len(attention_scores):  # ← FIXED: Added bounds check
            word = tokenizer.index_word.get(word_id, "<UNK>")
            score = attention_scores[idx]
            word_attention.append((word, score, idx))
    
    word_attention.sort(key=lambda x: x[1], reverse=True)
    return word_attention[:top_n]

# Analyze reviews
all_attention_data = []
for idx in range(min(100, len(X_test))):
    review_text = X_test.iloc[idx]
    attention_weights = attention_output[idx]
    actual_label = y_test.iloc[idx]
    predicted_label = y_pred[idx]
    
    important_words = get_important_words(review_text, attention_weights, tokenizer, top_n=5)
    
    if important_words:
        top_word, top_score, _ = important_words[0]
        all_attention_data.append({
            'review_idx': idx,
            'review_text': review_text[:100],
            'actual_label': actual_label,
            'predicted_label': predicted_label,
            'top_important_word': top_word,
            'top_attention_score': top_score,
            'top_5_words': ', '.join([w[0] for w in important_words])
        })

# Save to CSV
attention_df = pd.DataFrame(all_attention_data)
attention_df.to_csv("attention_analysis.csv", index=False)
print("✓ Attention analysis saved to: attention_analysis.csv")
print("\nSample Attention Analysis:")
print(attention_df.head(10))

print("="*60)




