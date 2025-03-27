This repository contains Python scripts for training a text classification model to classify chat messages as either male or female. The model is built using XGBoost for classification and Word2Vec for word embeddings. 
It also includes hyperparameter tuning using Stratified K-Fold cross-validation to optimize model performance.

Install required dependencies:
Create a requirements.txt 

Scripts
train.py (Train and Save the Model)
This script is used to train the classification model using Word2Vec embeddings for text data and XGBoost for classification. The script also performs hyperparameter tuning using Stratified K-Fold cross-validation.

Arguments:
training_data: The path to the folder containing male.txt and female.txt files.

model_file: The path to save the trained model.

The script first loads the male.txt and female.txt files from the training_data folder.

Word2Vec is used to convert text into word embeddings.

XGBoost is then trained using the Stratified K-Fold cross-validation technique to ensure balanced class distribution.

Hyperparameters like max_depth, reg_alpha, reg_lambda, n_estimators, and learning_rate are tuned using GridSearchCV.

The ROC AUC score and accuracy are computed and displayed.

classify.py (Classify a String Using the Model)
This script loads the trained model and classifies a string passed as an argument.

Arguments:
model_file: The trained model file (from train.py).

input_text: The input string to classify.

This will output either Male or Female based on the model's prediction.

Design Decisions
1. Libraries Used:
XGBoost: A tree-based model that provides high performance for classification tasks. It's used here for its ability to handle large datasets and produce fast predictions.

Word2Vec: A word embedding technique that converts text into vectors based on the context of words. It’s used to create meaningful representations of words in the text.

Scikit-learn: Used for Stratified K-Fold cross-validation and GridSearchCV for hyperparameter tuning.

2. Hyperparameters Tuned:
max_depth: Controls the maximum depth of trees in XGBoost.

reg_alpha: L1 regularization term on weights (helps with sparsity).

reg_lambda: L2 regularization term on weights.

n_estimators: Number of trees in the model.

learning_rate: Step size for the model's optimization.

3. Stratified K-Fold Cross-Validation:
Ensures that each fold has a balanced class distribution, which is crucial for imbalanced datasets.

Helps to evaluate the model performance more reliably by splitting the data into 5 different folds (default).

You can install the necessary Python dependencies by running the following:
pip install -r requirements.txt


Conclusion
This repository provides a simple yet effective solution for classifying text data into categories like male and female based on chat messages. By utilizing XGBoost and Word2Vec, it provides both speed and high accuracy, with the added benefit of hyperparameter tuning to optimize model performance.



