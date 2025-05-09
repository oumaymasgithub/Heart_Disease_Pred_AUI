import pandas as pd

# Load dataset
df = pd.read_csv("heart_disease_cleaveland.csv")

# Check for duplicates
duplicates = df.duplicated()  # Returns a boolean Series (True = Duplicate)
print("\n=== Duplicate Rows Count ===")
print(duplicates.sum())  # Count of duplicate rows
print("\n=== Duplicate Rows ===")
print(df[df.duplicated()])
df_cleaned = df.drop_duplicates()
print("\n=== Duplicates Removed ===")
print(df_cleaned.shape)  # New shape after removing duplicates
df_cleaned = df.drop_duplicates(subset=["patient_id"], keep="first")

#print("\n=== Duplicate Patient IDs ===")
#print(df[df.duplicated(subset=["id"])])  # Replace "patient_id" with the relevant column

