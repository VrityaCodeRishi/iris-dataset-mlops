import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
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
cv_results = {}
best_model = None
best_score = 0
best_cv_score = 0
best_name = None


for name, model in models.items():
    print(f"\nTraining {name}...")
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    cv_results[name] = {
        'mean': float(cv_mean),
        'std': float(cv_std),
        'scores': [float(s) for s in cv_scores]
    }
    print(f"  Cross-validation accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    results[name] = {
        'test_accuracy': float(test_score),
        'cv_mean': float(cv_mean),
        'cv_std': float(cv_std)
    }
    print(f"  Test set accuracy: {test_score:.4f}")
    print(f"  Test set size: {len(y_test)} samples")
    print(f"  Correct predictions: {sum(y_pred == y_test)}/{len(y_test)}")
    
    if cv_mean > best_cv_score:
        best_cv_score = cv_mean
        best_score = test_score
        best_name = name
        best_model = model

print(f"\n{'='*50}")
print(f"Best model: {best_name}")
print(f"  Cross-validation accuracy: {best_cv_score:.4f}")
print(f"  Test set accuracy: {best_score:.4f}")
print(f"{'='*50}")


print(f"\nRetraining {best_name} on full dataset...")
best_model.fit(X, y)


with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('results.json', 'w') as f:
    json.dump({
        'best_model': best_name,
        'best_test_accuracy': float(best_score),
        'best_cv_accuracy': float(best_cv_score),
        'all_results': results,
        'cv_results': cv_results
    }, f, indent=2)

print(f"Model saved as model.pkl")
