import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

# --- Load Cleveland dataset ---
cleveland = pd.read_csv("cleavland/heart_disease_train.csv")
cleveland['target'] = cleveland['target'].apply(lambda x: 1 if x > 0 else 0)

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

# Extract features and labels
Xc = cleveland[clinical_features]
Xl = cleveland[lifestyle_features]
y_clev = cleveland['target']

# --- Load Cardiovascular dataset ---
cardio = pd.read_csv("cardiovascular/cardio_train_preprocessed.csv", sep=';')
X_cardio = cardio.drop(columns=['id', 'cardio'])  # ✅ Drop 'id' column
y_cardio = cardio['cardio']

# --- Load trained base models ---
model_clinical = joblib.load("cleavland/model_clinical.pkl")
model_lifestyle = joblib.load("cleavland/model_lifestyle.pkl")
model_cardio = joblib.load("cardiovascular/model_cardio.pkl")

# --- Generate base model predictions ---
clev_prob_1 = model_clinical.predict_proba(Xc)[:, 1]
clev_prob_2 = model_lifestyle.predict_proba(Xl)[:, 1]
clev_avg = (clev_prob_1 + clev_prob_2) / 2

cardio_prob = model_cardio.predict_proba(X_cardio)[:, 1]

# --- Align lengths (smallest dataset size wins) ---
min_len = min(len(clev_avg), len(cardio_prob))
meta_X = np.column_stack((clev_avg[:min_len], cardio_prob[:min_len]))
meta_y = y_clev[:min_len]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    meta_X, meta_y, test_size=0.2, stratify=meta_y, random_state=42
)

# --- Train fusion model ---
fusion_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
fusion_model.fit(X_train, y_train)

# --- Evaluate ---
fusion_preds = fusion_model.predict(X_test)
fusion_probs = fusion_model.predict_proba(X_test)[:, 1]

print("\nFusion Meta-Model Evaluation (Test Set):")
print(classification_report(y_test, fusion_preds))
print("ROC AUC:", roc_auc_score(y_test, fusion_probs))

# --- Save final model ---
joblib.dump(fusion_model, "fusion_meta_model.pkl")
print("\nSaved: fusion_meta_model.pkl")
