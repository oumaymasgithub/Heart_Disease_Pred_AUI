import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load cardio dataset
df = pd.read_csv("cardio_train_preprocessed.csv", sep=';')

# Split features/target
X = df.drop(columns=['id', 'cardio'])
y = df['cardio']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Training features shape:", X_train.shape)
print("Columns used:", X_train.columns.tolist())
# Train model
model_cardio = RandomForestClassifier(n_estimators=100, random_state=42)
model_cardio.fit(X_train, y_train)

# Evaluate
probs = model_cardio.predict_proba(X_test)[:, 1]
preds = model_cardio.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, preds))
print(" ROC AUC Score:", roc_auc_score(y_test, probs))

# Save model
joblib.dump(model_cardio, "model_cardio.pkl")
