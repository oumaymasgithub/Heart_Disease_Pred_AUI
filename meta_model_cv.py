import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("heart_disease_train.csv")
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# One-hot columns (already applied in your dataset)
clinical_features = ['resting bp s', 'cholesterol', 'resting ecg_1', 'resting ecg_2', 'max heart rate', 'oldpeak', 'ST slope_1', 'ST slope_2', 'ST slope_3']
lifestyle_features = ['age', 'sex', 'fasting blood sugar', 'exercise angina', 'chest pain type_2', 'chest pain type_3', 'chest pain type_4']

Xc = df[clinical_features]
Xl = df[lifestyle_features]
y = df['target']

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(Xc, y), 1):
    print(f"\nFold {fold}:")

    # Split for this fold
    Xc_train, Xc_test = Xc.iloc[train_idx], Xc.iloc[test_idx]
    Xl_train, Xl_test = Xl.iloc[train_idx], Xl.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train base models
    model_clinical = LogisticRegression(max_iter=1000)
    model_clinical.fit(Xc_train, y_train)
    pred_c_train = model_clinical.predict_proba(Xc_train)[:, 1]
    pred_c_test = model_clinical.predict_proba(Xc_test)[:, 1]

    model_lifestyle = RandomForestClassifier(n_estimators=100, random_state=42)
    model_lifestyle.fit(Xl_train, y_train)
    pred_l_train = model_lifestyle.predict_proba(Xl_train)[:, 1]
    pred_l_test = model_lifestyle.predict_proba(Xl_test)[:, 1]

    # Meta model input
    meta_train = np.column_stack((pred_c_train, pred_l_train))
    meta_test = np.column_stack((pred_c_test, pred_l_test))

    # Train meta model
    meta_model = XGBClassifier(eval_metric='logloss')
    meta_model.fit(meta_train, y_train)

    # Predict and evaluate
    preds = meta_model.predict(meta_test)
    probs = meta_model.predict_proba(meta_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    auc_scores.append(auc)

    print(classification_report(y_test, preds))
    print(f"ROC AUC for Fold {fold}: {auc:.4f}")

# Final average score
print("\nCross-Validation AUC Scores:", [f"{s:.4f}" for s in auc_scores])
print(f"Mean ROC AUC: {np.mean(auc_scores):.4f}")



# Plot the ROC AUC scores from cross-validation
plt.figure(figsize=(8, 5))
plt.boxplot(auc_scores, vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))

# Scatter plot of individual fold AUCs
plt.scatter(auc_scores, [1]*len(auc_scores), color='red', zorder=3, label='Fold AUCs')

# Labels and title
plt.title("Cross-Validation ROC AUC Scores for Meta-Model", fontsize=14)
plt.xlabel("ROC AUC Score")
plt.yticks([1], ['Meta-Model (Stacked)'])
plt.legend()
plt.grid(True, axis='x')

# Show the plot
plt.tight_layout()
plt.show()
