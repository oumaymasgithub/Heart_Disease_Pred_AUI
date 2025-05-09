import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load training data to re-train base models
train_df = pd.read_csv("heart_disease_train.csv")
train_df['target'] = train_df['target'].apply(lambda x: 1 if x > 0 else 0)

# Load new patients
new_patients_df = pd.read_csv("new_patients.csv")

# Feature groups
clinical_features = ['resting bp s', 'cholesterol', 'resting ecg', 'max heart rate', 'oldpeak', 'ST slope']
lifestyle_features = ['age', 'sex', 'chest pain type', 'fasting blood sugar', 'exercise angina']

# Split training data
Xc_train = train_df[clinical_features]
Xl_train = train_df[lifestyle_features]
y_train = train_df['target']

# Base model 1
model_clinical = LogisticRegression(max_iter=1000)
model_clinical.fit(Xc_train, y_train)

# Base model 2
model_lifestyle = RandomForestClassifier(n_estimators=100, random_state=42)
model_lifestyle.fit(Xl_train, y_train)

# Meta model
meta_train = np.column_stack((
    model_clinical.predict_proba(Xc_train)[:, 1],
    model_lifestyle.predict_proba(Xl_train)[:, 1]
))

meta_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
meta_model.fit(meta_train, y_train)

# Predictions for new patients
Xc_new = new_patients_df[clinical_features]
Xl_new = new_patients_df[lifestyle_features]

pred_clinical_new = model_clinical.predict_proba(Xc_new)[:, 1]
pred_lifestyle_new = model_lifestyle.predict_proba(Xl_new)[:, 1]

meta_input_new = np.column_stack((pred_clinical_new, pred_lifestyle_new))

# Predict risk
probs = meta_model.predict_proba(meta_input_new)[:, 1]
preds = meta_model.predict(meta_input_new)

# Output
for i in range(len(new_patients_df)):
    print(f"\nPatient {i+1}")
    print(f"Predicted Risk: {'High Risk' if preds[i] == 1 else 'Low Risk'} ({probs[i]*100:.1f}%)")

# SHAP Force Plot
explainer = shap.Explainer(meta_model)
shap_values = explainer(meta_input_new)

for i in range(len(new_patients_df)):
    print(f"\n--- SHAP Explanation for Patient {i+1} ---")
    shap.plots.force(shap_values[i], matplotlib=True, feature_names=["ClinicalModel", "LifestyleModel"])
