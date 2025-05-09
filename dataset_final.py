import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from cryptography.fernet import Fernet
import io

# 1 Load the Dataset
dataset_path = "heart_disease_cleaveland.csv"
df = pd.read_csv(dataset_path)

# Display dataset overview
print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Summary")
print(df.info())

# 2 Handle Missing Values
print("\nMissing Values in Each Column")
print(df.isnull().sum())

df.fillna(df.median(), inplace=True)  # Fill missing values with column median
print("\nMissing values handled.")

# 3 Detect and Remove Duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows Count: {duplicates}")

df.drop_duplicates(inplace=True)
print("\nDuplicates removed.")

# 4 Convert Data Types
print("\nData Types Before Conversion")
print(df.dtypes)

# Convert categorical columns to category type
categorical_cols = ["sex", "exercise angina", "chest pain type", "resting ecg", "ST slope"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

# Convert date columns (if applicable)
date_cols = ["diagnosis_date"]  # Replace with actual column name
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        print(f"Converted {col} to datetime")

print("\nData Types After Conversion")
print(df.dtypes)


# 5 Handle Outliers (IQR Method)
def remove_outliers_iqr(df, columns, threshold=1.5):
    """Removes outliers using the IQR (Interquartile Range) method."""
    for col in columns:
        if df[col].dtype in ["int64", "int32", "int16", "float64", "float32"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            print(f"{col}: Removed {outliers} outliers")

            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df


df = remove_outliers_iqr(df, df.select_dtypes(include=["int64", "float64"]).columns)
print("\nOutliers removed.")

# 6 Label Encoding for Binary Categories
binary_cols = ["sex", "exercise angina"]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

print("\nLabel Encoding applied.")

# 7 One-Hot Encoding for Multiclass Categories
multi_class_cols = ["chest pain type", "resting ecg", "ST slope"]
df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)

print("\nOne-Hot Encoding applied.")

# 8 Train-Test Split (80-20)
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=42)

print("\nDataset Splitting Results")
print(f"Total dataset shape: {df.shape}")
print(f"Training set shape: {df_train.shape}")
print(f"Testing set shape: {df_test.shape}")

# Save train-test datasets
df_train.to_csv("heart_disease_train.csv", index=False)
df_test.to_csv("heart_disease_test.csv", index=False)

# 9 Encrypt the Dataset
# Generate and save encryption key (only once)
key = Fernet.generate_key()
with open("encryption_key.key", "wb") as key_file:
    key_file.write(key)

print("\nEncryption key generated and saved as 'encryption_key.key'")

# Convert DataFrame to CSV string
csv_data = df.to_csv(index=False)

# Encrypt CSV data
fernet = Fernet(key)
encrypted_data = fernet.encrypt(csv_data.encode())

# Save encrypted data
with open("heart_disease_encrypted.enc", "wb") as encrypted_file:
    encrypted_file.write(encrypted_data)

print("\nDataset encrypted and saved as 'heart_disease_encrypted.enc'")

# 10 Decrypt Dataset (When Needed)
# Load encryption key
with open("encryption_key.key", "rb") as key_file:
    key = key_file.read()

fernet = Fernet(key)

# Read encrypted data
with open("heart_disease_encrypted.enc", "rb") as encrypted_file:
    encrypted_data = encrypted_file.read()

# Decrypt the data
decrypted_data = fernet.decrypt(encrypted_data).decode()

# Convert decrypted CSV string back to DataFrame
df_decrypted = pd.read_csv(io.StringIO(decrypted_data))

# Save decrypted dataset
df_decrypted.to_csv("heart_disease_decrypted.csv", index=False)

print("\nDataset decrypted and saved as 'heart_disease_decrypted.csv'")
