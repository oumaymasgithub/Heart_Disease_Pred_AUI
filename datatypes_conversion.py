import pandas as pd
import numpy as np

# Load dataset
dataset_path = "heart_disease_cleaveland.csv"  # Ensure the file is in the working directory
df = pd.read_csv(dataset_path)

# Check Current Data Types

print("\n=== Data Types Before Conversion ===")
print(df.dtypes)

# Convert Numeric Columns

# Identify numeric columns stored as "object" (string)
object_cols = df.select_dtypes(include=["object"]).columns

for col in object_cols:
    try:
        df[col] = pd.to_numeric(df[col], errors="raise")  # Convert to numeric if possible
        print(f"Converted {col} to numeric")
    except ValueError:
        print(f"Skipped {col}, still categorical")

# convert Categorical Columns


# Identify categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].astype("category")  # Convert text-based categories
    print(f"Converted {col} to category")

# Convert Date Columns

# Identify possible date columns (Modify column names if necessary)
date_cols = ["diagnosis_date"]  # Replace with actual date column names

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")  # Convert date strings to datetime
        print(f"Converted {col} to datetime")

# Optimize Memory Usage

# Convert large integers to lower precision
for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int16")

# Convert floating-point numbers to lower precision
for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")

print("\n=== Data Types After Conversion ===")
print(df.dtypes)


#  Save the Converted Dataset

converted_path = "heart_disease_converted.csv"
df.to_csv(converted_path, index=False)
print(f"\nConverted dataset saved as: {converted_path}")
