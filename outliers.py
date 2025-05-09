import pandas as pd
import numpy as np

# Load dataset
dataset_path = "heart_disease_cleaveland.csv"  # Ensure the file is in the working directory
df = pd.read_csv(dataset_path)

#  Check Current Data Types

print("\n=== Data Types Before Conversion ===")
print(df.dtypes)


#  Convert Numeric Columns

# Identify numeric columns stored as "object" (string)
object_cols = df.select_dtypes(include=["object"]).columns

for col in object_cols:
    try:
        df[col] = pd.to_numeric(df[col], errors="raise")  # Convert to numeric if possible
        print(f"Converted {col} to numeric")
    except ValueError:
        print(f"Skipped {col}, still categorical")

# Convert Categorical Columns

# Identify categorical columns

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


# Detect and Handle Outliers (IQR Method)

def remove_outliers_iqr(df, columns, threshold=1.5):
    """
    Removes outliers using the IQR (Interquartile Range) method.
    Any value outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] is considered an outlier.
    """
    for col in columns:
        if df[col].dtype in ["int64", "int32", "int16", "float64", "float32"]:  # Apply only to numerical columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            print(f"{col}: Removed {outliers} outliers")

            # Remove outliers
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df


# Apply outlier removal
num_cols = df.select_dtypes(include=["int64", "int32", "int16", "float64", "float32"]).columns
df = remove_outliers_iqr(df, num_cols)

print("\nOutliers removed successfully.")

# Optimize Memory Usage

# Convert large integers to lower precision
for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int16")

# Convert floating-point numbers to lower precision
for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")

print("\n=== Data Types After Conversion ===")
print(df.dtypes)


# Save the Cleaned Dataset

converted_path = "heart_disease_cleaned.csv"
df.to_csv(converted_path, index=False)
print(f"\nCleaned dataset saved as: {converted_path}")
