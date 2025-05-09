import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset_path = "heart_disease_cleaveland.csv"
df = pd.read_csv(dataset_path)

# Label Encoding for Binary Categories

binary_cols = ["sex", "exercise angina"]  # Columns with binary values

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])  # Converts 0/1 categorical values into numerical

print(f"Label Encoding applied to: {binary_cols}")

#  One-Hot Encoding for Multiclass Categories


multi_class_cols = ["chest pain type", "resting ecg", "ST slope"]

df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)  # Avoids multicollinearity

print(f"One-Hot Encoding applied to: {multi_class_cols}")

#  Verify Encoding Results

print("\n=== Data After Encoding ===")
print(df.head())

# Save the processed dataset
df.to_csv("heart_disease_encoded.csv", index=False)
print("\nEncoded dataset saved as 'heart_disease_encoded.csv'")
