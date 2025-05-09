import pandas as pd
from sklearn.model_selection import train_test_split


def load_preprocessed_data(filepath):
    """
    Load the preprocessed dataset from the given filepath.
    """
    df = pd.read_csv(filepath)
    return df


def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataframe into training and testing sets.

    Parameters:
    - df: pandas DataFrame, the input data
    - test_size: float, the proportion of the dataset to include in the test split
    - random_state: int, seed used by the random number generator for reproducibility

    Returns:
    - train: DataFrame containing the training set
    - test: DataFrame containing the test set
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test


def main():
    # Path to the preprocessed dataset
    preprocessed_filepath = "cardiovascular/cardio_train_preprocessed.csv"

    # Load the preprocessed dataset
    df = load_preprocessed_data(preprocessed_filepath)
    print("Preprocessed dataset loaded:")
    print(df.head())

    # Split the data into training and testing sets (80/20 split)
    train_df, test_df = split_data(df, test_size=0.2, random_state=42)

    print("\nTraining set preview:")
    print(train_df.head())
    print("\nTesting set preview:")
    print(test_df.head())

    # Save the training and testing sets to CSV files
    train_output_filepath = "cardiovascular/cardio_train_split_train.csv"
    test_output_filepath = "cardiovascular/cardio_train_split_test.csv"

    train_df.to_csv(train_output_filepath, index=False)
    test_df.to_csv(test_output_filepath, index=False)

    print(f"\nTraining set saved to: {train_output_filepath}")
    print(f"Testing set saved to: {test_output_filepath}")


if __name__ == "__main__":
    main()
