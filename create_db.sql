CREATE DATABASE IF NOT EXISTS heartdb;
USE heartdb;

DROP TABLE IF EXISTS patients;

CREATE TABLE patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    sex INT,
    cp INT,
    trestbps INT,
    chol INT,
    fbs INT,
    restecg INT,
    thalach INT,
    exang INT,
    oldpeak FLOAT,
    slope INT,
    ca INT,
    thal INT,
    gender INT,
    height INT,
    weight INT,
    ap_hi INT,
    ap_lo INT,
    cholesterol_level INT,
    gluc INT,
    smoke INT,
    alco INT,
    active INT,
    prediction VARCHAR(50),
    probability FLOAT,
    date DATETIME
);
