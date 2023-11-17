from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import json
import numpy as np
import pandas as pd

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define MLPClassifier with parameter grid
clf = MLPClassifier(max_iter=500, random_state=0)
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# Grid search with cross-validation
clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)

# Best model
best_model = clf.best_estimator_

# Get overall accuracy
acc = best_model.score(X_test, y_test)

# Get precision and recall
y_score = best_model.predict(X_test)
prec = precision_score(y_test, y_score)
rec = recall_score(y_test, y_score)

# Get the loss
loss = best_model.loss_curve_
pd.DataFrame(loss, columns=["loss"]).to_csv("loss.csv", index=False)

with open("metrics.json", 'w') as outfile:
        json.dump({"accuracy": acc, "precision": prec, "recall": rec}, outfile)
