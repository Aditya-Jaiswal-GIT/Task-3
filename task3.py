"""
Titanic Survival Prediction
---------------------------------
• Data file : Document from Aditya   (copy/rename to titanic.csv if you like)
• Model     : Logistic Regression
• Metrics   : Accuracy, Precision, Recall, F1, ROC-AUC
• Plots     : Confusion-matrix heat-map, ROC curve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score,
                             RocCurveDisplay)

# ---------- 1. LOAD DATA ----------------------------------------------------
df = pd.read_csv('Titanic-Dataset.csv')        # adjust path as needed

# ---------- 2. BASIC CLEANING / PREP ----------------------------------------
# Drop columns with too many categories or high missingness
df = df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# One-hot encode categorical vars (drop_first avoids dummy trap)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# ---------- 3. TRAIN–TEST SPLIT --------------------------------------------
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------- 4. MODEL: LOGISTIC REGRESSION -----------------------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(
        max_iter=1000,
        class_weight='balanced'   # combats slight class imbalance
    ))
])

pipe.fit(X_train, y_train)

# ---------- 5. EVALUATION ---------------------------------------------------
y_pred  = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred, digits=4))

print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# ----- Confusion-Matrix Heat-map (matplotlib only) -----
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4,4))
im = ax.imshow(cm, cmap='Blues')

# Axis labels & ticks
classes = ['Not Survived', 'Survived']
ax.set(
    xticks=np.arange(len(classes)),
    yticks=np.arange(len(classes)),
    xticklabels=classes, yticklabels=classes,
    ylabel='True label', xlabel='Predicted label',
    title='Confusion Matrix'
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Annotate cells
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
plt.show()

# ----- ROC Curve -----
RocCurveDisplay.from_estimator(pipe, X_test, y_test)
plt.title('ROC Curve')
plt.show()