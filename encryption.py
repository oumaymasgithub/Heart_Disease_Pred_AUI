import pandas as pd
from cryptography.fernet import Fernet
import io


# Generate Encryption Key (Only Once)

# Generate a new key (do this once and keep the key safe!)
key = Fernet.generate_key()

# Save key to a file
with open("encryption_key.key", "wb") as key_file:
    key_file.write(key)

print("\n Encryption key generated and saved as 'encryption_key.key'")

# Encrypt Dataset

# Load dataset
dataset_path = "heart_disease_cleaveland.csv"
df = pd.read_csv(dataset_path)

# Convert DataFrame to CSV format (string)
csv_data = df.to_csv(index=False)

# Initialize Fernet with the key
fernet = Fernet(key)

# Encrypt CSV data
encrypted_data = fernet.encrypt(csv_data.encode())

# Save encrypted data to a file
with open("heart_disease_encrypted.enc", "wb") as encrypted_file:
    encrypted_file.write(encrypted_data)

print("\n dataset encrypted and saved as 'heart_disease_encrypted.enc'")

# Decrypt Dataset (When Needed)

# Load encryption key
with open("encryption_key.key", "rb") as key_file:
    key = key_file.read()

# Initialize Fernet with the same key
fernet = Fernet(key)

# Read encrypted data
with open("heart_disease_encrypted.enc", "rb") as encrypted_file:
    encrypted_data = encrypted_file.read()

# Decrypt the data
decrypted_data = fernet.decrypt(encrypted_data).decode()

# Convert decrypted CSV string back to DataFrame
df_decrypted = pd.read_csv(io.StringIO(decrypted_data))

# Save decrypted file
df_decrypted.to_csv("heart_disease_decrypted.csv", index=False)

print("\n Dataset decrypted and saved as 'heart_disease_decrypted.csv'")
