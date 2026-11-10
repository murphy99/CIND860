from nltk.corpus import stopwords
import nltk

import time

import numpy as np
import pandas as pd

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#print("Loading data..."):whi

#train = pd.read_csv(f"{DATA_DIR}/imdb_train.csv")
# ... rest of data loading ...

GLOVE_PATH = "data/glove/glove.6B.100d.txt"

print("\nLoading GloVe embeddings (filtering stop words)...")
t0 = time.time()
embeddings_index = {}
with open(GLOVE_PATH, encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        
        # ← FILTER OUT STOP WORDS
        if word.lower() in stop_words:
            continue
        
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

print(f"Loaded {len(embeddings_index)} word vectors ({time.time() - t0:.1f}s)")
print(f"(Filtered out {len(stop_words)} stop words)")

# Rest of your code continues as normal...
# The model will be trained with filtered embeddings