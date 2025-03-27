import sys
import pickle
import numpy as np
from gensim.models import Word2Vec

# Check command-line arguments
if len(sys.argv) < 3:
    print("Usage: ./classify.py model_file 'input text'")
    sys.exit(1)

model_file = sys.argv[1]
input_text = sys.argv[2]

# Load trained classifier
try:
    with open(model_file, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file '{model_file}' not found!")
    sys.exit(1)

# Load trained Word2Vec model
try:
    w2v_model = Word2Vec.load("word2vec.model")
except FileNotFoundError:
    print("Error: Word2Vec model 'word2vec.model' not found! Run training first.")
    sys.exit(1)

# Convert input text to Word2Vec vector
tokens = input_text.split()

def vectorize_text(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

vector = vectorize_text(tokens, w2v_model).reshape(1, -1)

# Predict gender
predicted_label = model.predict(vector)[0]
label_map = {0: "Male", 1: "Female"}

print(label_map[predicted_label])
