from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
dataset_path = "heart_disease_cleaveland.csv"
df = pd.read_csv(dataset_path)

# Train-Test Split (80% Train, 20% Test)
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=42)

# Print dataset sizes
print("\n=== Dataset Splitting Results ===")
print(f"Total dataset shape: {df.shape}")
print(f"Training set shape: {df_train.shape}")
print(f"Testing set shape: {df_test.shape}")

# Save the split datasets
df_train.to_csv("heart_disease_train.csv", index=False)
df_test.to_csv("heart_disease_test.csv", index=False)

print("\nTrain & Test datasets saved as 'heart_disease_train.csv' & 'heart_disease_test.csv'")
