from flask import Flask, render_template, request
import joblib
import mysql.connector
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load models
model_clinical = joblib.load("models/model_clinical.pkl")
model_lifestyle = joblib.load("models/model_lifestyle.pkl")
model_cardio = joblib.load("models/model_cardio.pkl")
fusion_meta_model = joblib.load("models/fusion_meta_model.pkl")

# MySQL connection config
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'bebeetch1',
    'database': 'heart_predictions'
}

@app.route('/')
def form():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY date DESC")
    rows = cursor.fetchall()
    conn.close()
    return render_template("form.html", records=rows)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    name = data['name']

    # Model input fields
    clinical_fields = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'ca', 'thal']
    lifestyle_fields = ['age', 'sex', 'cp', 'exang', 'oldpeak', 'slope', 'fbs']
    cardio_fields = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active']

    # Prepare model inputs
    X_clinical = np.array([[float(data[field]) for field in clinical_fields]])
    X_lifestyle = np.array([[float(data[field]) for field in lifestyle_fields]])
    X_cardio = np.array([[float(data[field]) for field in cardio_fields]])

    # Predictions
    pred_clinical = model_clinical.predict_proba(X_clinical)[:, 1][0]
    pred_lifestyle = model_lifestyle.predict_proba(X_lifestyle)[:, 1][0]
    pred_cardio = model_cardio.predict_proba(X_cardio)[:, 1][0]

    fusion_input = np.array([[(pred_clinical + pred_lifestyle) / 2, pred_cardio]])
    prob = fusion_meta_model.predict_proba(fusion_input)[:, 1][0]
    result = "High Risk" if prob > 0.5 else "Low Risk"

    # DB values
    db_data = [
        float(data['age']), int(data['sex']), int(data['cp']), int(data['trestbps']),
        int(data['chol']), int(data['fbs']), int(data['restecg']), int(data['thalach']),
        int(data['exang']), float(data['oldpeak']), int(data['slope']), int(data['ca']),
        int(data['thal']), int(data['gender']), int(data['height']), int(data['weight']),
        int(data['ap_hi']), int(data['ap_lo']), int(data['cholesterol']),
        int(data['gluc']), int(data['smoke']), int(data['alco']), int(data['active'])
    ]

    # Insert into DB
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO patients (
            name, age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal,
            gender, height, weight, ap_hi, ap_lo, cholesterol_level,
            glucose_level, smoker, alco, active,
            prediction, probability, date
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s)
    ''', (name, *db_data, result, float(prob), datetime.now()))
    conn.commit()
    conn.close()

    return render_template("result.html", result=result, prob=f"{prob:.2%}")

@app.route('/records')
def records():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY date DESC")
    rows = cursor.fetchall()
    conn.close()

    headers = ["ID", "Name", "Age", "Sex", "CP", "Trestbps", "Chol", "FBS", "RestECG", "Thalach", "Exang",
               "Oldpeak", "Slope", "CA", "Thal", "Gender", "Height", "Weight", "Ap_hi", "Ap_lo",
               "Cholesterol", "Glucose", "Smoker", "Alcohol", "Active", "Prediction", "Probability", "Date"]

    html = "<h2>Saved Patient Predictions</h2><table border='1'><tr>"
    for h in headers:
        html += f"<th>{h}</th>"
    html += "</tr>"

    for row in rows:
        html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    html += "</table><br><a href='/'>Back to form</a>"
    return html

if __name__ == '__main__':
    app.run(debug=True)
