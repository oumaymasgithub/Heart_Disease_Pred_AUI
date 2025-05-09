import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import shap
import joblib
import matplotlib.pyplot as plt

# Load preprocessed train/test data
train_df = pd.read_csv("heart_disease_train.csv")
test_df = pd.read_csv("heart_disease_test.csv")

print("Columns in train_df:", train_df.columns.tolist())

# Ensure target is binary
train_df['target'] = train_df['target'].apply(lambda x: 1 if x > 0 else 0)
test_df['target'] = test_df['target'].apply(lambda x: 1 if x > 0 else 0)

# Define feature groups
clinical_features = [
    'resting bp s', 'cholesterol',
    'resting ecg_1', 'resting ecg_2',
    'max heart rate', 'oldpeak',
    'ST slope_1', 'ST slope_2', 'ST slope_3'
]

lifestyle_features = [
    'age', 'sex', 'fasting blood sugar', 'exercise angina',
    'chest pain type_2', 'chest pain type_3', 'chest pain type_4'
]

# Split into X and y
Xc_train = train_df[clinical_features]
Xl_train = train_df[lifestyle_features]
y_train = train_df['target']

Xc_test = test_df[clinical_features]
Xl_test = test_df[lifestyle_features]
y_test = test_df['target']

# Train base models
model_clinical = LogisticRegression(max_iter=1000)
model_clinical.fit(Xc_train, y_train)
pred_clinical_train = model_clinical.predict_proba(Xc_train)[:, 1]
pred_clinical_test = model_clinical.predict_proba(Xc_test)[:, 1]

model_lifestyle = RandomForestClassifier(n_estimators=100, random_state=42)
model_lifestyle.fit(Xl_train, y_train)
pred_lifestyle_train = model_lifestyle.predict_proba(Xl_train)[:, 1]
pred_lifestyle_test = model_lifestyle.predict_proba(Xl_test)[:, 1]

# Save base models in current folder (cleavland/)
joblib.dump(model_clinical, 'model_clinical.pkl')
print("Saved: model_clinical.pkl")

joblib.dump(model_lifestyle, 'model_lifestyle.pkl')
print("Saved: model_lifestyle.pkl")

# Create meta features and train meta-model
meta_train = np.column_stack((pred_clinical_train, pred_lifestyle_train))
meta_test = np.column_stack((pred_clinical_test, pred_lifestyle_test))

meta_model = XGBClassifier(eval_metric='logloss')
meta_model.fit(meta_train, y_train)

# Evaluate meta-model
meta_preds = meta_model.predict(meta_test)
meta_probs = meta_model.predict_proba(meta_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, meta_preds))
print("ROC AUC Score:", roc_auc_score(y_test, meta_probs))

# SHAP Explanation
explainer = shap.Explainer(meta_model)
shap_values = explainer(meta_test)
shap.summary_plot(shap_values, features=meta_test, feature_names=["ClinicalModel", "LifestyleModel"])
