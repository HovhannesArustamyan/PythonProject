import os
import sys
import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

# Check command-line arguments
if len(sys.argv) < 3:
    print("Usage: ./train.py training_data model_file")
    sys.exit(1)

training_data_folder = sys.argv[1]
model_file = sys.argv[2]

# Load text files
def load_texts(file_path, label):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        return [(line.strip(), label) for line in f.readlines() if line.strip()]

male_texts = load_texts(os.path.join(training_data_folder, "male.txt"), 0)
female_texts = load_texts(os.path.join(training_data_folder, "female.txt"), 1)

# Create DataFrame
data = pd.DataFrame(male_texts + female_texts, columns=["text", "label"])

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Tokenize text (split into words)
data["tokens"] = data["text"].apply(lambda x: x.split())

# Train Word2Vec model
w2v_model = Word2Vec(sentences=data["tokens"], vector_size=100, window=5, min_count=1, workers=4)
w2v_model.save("word2vec.model")

# Convert text to Word2Vec embeddings
def vectorize_text(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

data["vector"] = data["tokens"].apply(lambda x: vectorize_text(x, w2v_model))

# Prepare training data
X = np.vstack(data["vector"].values)
y = data["label"].values

# Set up Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define hyperparameters to tune
param_grid = {
    'max_depth': [3, 5, 7],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1],
}

# Set up the XGBoost model
model = XGBClassifier(random_state=42)

# Set up GridSearchCV with StratifiedKFold
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='roc_auc', cv=kf, verbose=1, n_jobs=-1)

# Perform hyperparameter tuning and cross-validation
grid_search.fit(X, y)

# Best hyperparameters and model evaluation
print("Best Hyperparameters:", grid_search.best_params_)

# Use the best model from GridSearchCV to predict
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X)
y_prob = best_model.predict_proba(X)[:, 1]  # Get probabilities for ROC AUC calculation

# Compute accuracy and ROC AUC score
accuracy = accuracy_score(y, y_pred)
roc_auc = roc_auc_score(y, y_prob)

print(f"Training Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# Save the trained model
with open(model_file, "wb") as f:
    pickle.dump(best_model, f)

print(f"Training complete! Model saved as '{model_file}'.")
