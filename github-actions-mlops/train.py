import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
import os

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "logistic_regression": LogisticRegression(random_state=42, max_iter=200),
    "random_forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "svm": SVC(random_state=42, probability=True)
}

results = {}
best_model = None
best_score = 0
best_name = None

# Train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    results[name] = score
    print(f"{name}: {score:.4f}")

    if score > best_score:
        best_score = score
        best_name = name
        best_model = model

print(f"\nBest model: {best_name} with accuracy {best_score:.4f}")

# Retrain best model on full dataset
print(f"\nRetraining {best_name} on full dataset...")
best_model.fit(X, y)

# Save the best model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save results
with open('results.json', 'w') as f:
    json.dump({
        'best_model': best_name,
        'best_accuracy': float(best_score),
        'all_results': results
    }, f, indent=2)

print(f"Model saved as model.pkl")
