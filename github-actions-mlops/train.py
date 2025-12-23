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


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "svm": SVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}: {accuracy_score(y_test, y_pred)}")
    print(f"{name}: {classification_report(y_test, y_pred)}")
    pickle.dump(model, open(f"models/{name}.pkl", "wb"))

results = {}
best_model = None
best_score = 0
best_name = None

for name, model in models.items():
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)
    if results[name] > best_score:
        best_score = results[name]
        best_name = name
        best_model = model

print(f"Best model: {best_name} with accuracy {best_score}")
pickle.dump(best_model, open(f"models/{best_name}.pkl", "wb"))
