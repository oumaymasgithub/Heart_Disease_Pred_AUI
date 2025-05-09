import pandas as pd
import numpy as np
import joblib

# --- Load models ---
model_clinical = joblib.load("cleavland/model_clinical.pkl")
model_lifestyle = joblib.load("cleavland/model_lifestyle.pkl")
model_cardio = joblib.load("cardiovascular/model_cardio.pkl")
fusion_model = joblib.load("fusion_meta_model.pkl")

# --- Define feature groups used by each base model ---
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

cardio_features = [
    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]

# --- Load new patient data ---
new_clev = pd.read_csv("cleavland/new_heart_disease_patients.csv")  # Cleveland-like
new_cardio = pd.read_csv("cardiovascular/new_cardio_patients.csv", sep=';')  # Cardio-like

# --- Prediction for one patient ---
while True:
    try:
        patient_idx = int(input(f"\nEnter patient index (0 to {len(new_clev)-1}) or -1 to exit: "))
        if patient_idx == -1:
            print("Exiting...")
            break

        if patient_idx < 0 or patient_idx >= len(new_clev):
            print("Invalid index. Try again.")
            continue

        # Extract one patient
        clev_patient_clinical = new_clev.loc[patient_idx, clinical_features].to_frame().T
        clev_patient_lifestyle = new_clev.loc[patient_idx, lifestyle_features].to_frame().T
        cardio_patient = new_cardio.loc[patient_idx, cardio_features].to_frame().T

        # Get predictions from base models
        clev_prob_1 = model_clinical.predict_proba(clev_patient_clinical)[:, 1]
        clev_prob_2 = model_lifestyle.predict_proba(clev_patient_lifestyle)[:, 1]
        cardio_prob = model_cardio.predict_proba(cardio_patient)[:, 1]

        # Stack and predict with fusion model
        stacked_features = np.column_stack(((clev_prob_1 + clev_prob_2) / 2, cardio_prob))
        final_prob = fusion_model.predict_proba(stacked_features)[:, 1]
        final_pred = fusion_model.predict(stacked_features)

        print("\nPrediction for Patient", patient_idx)
        print(f"Predicted Risk: {'High Risk' if final_pred[0] == 1 else 'Low Risk'} ({final_prob[0]*100:.1f}%)")

    except Exception as e:
        print(f"Error: {e}. Please try again.")
