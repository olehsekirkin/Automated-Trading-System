import pandas as pd

def preprocess_stock_data(file_path):
    # Read the .csv file into a DataFrame
    df = pd.read_csv(file_path)

    # Display the original column names
    print("Original Column Names:")
    print(df.columns)

    # Count the number of rows before preprocessing
    original_rows = df.shape[0]

    # Drop any duplicate rows
    df.drop_duplicates(inplace=True)
    print("\nDuplicate rows removed.")

    # Count the number of rows after removing duplicates
    rows_after_duplicates = df.shape[0]
    print(f"Rows removed: {original_rows - rows_after_duplicates}")

    # Count the number of missing values before preprocessing
    missing_values_before = df.isnull().sum().sum()

    # Drop any missing values
    df.dropna(inplace=True)
    print("Missing values removed.")

    # Count the number of missing values after removing missing values
    missing_values_after = df.isnull().sum().sum()
    print(f"Missing values removed: {missing_values_before - missing_values_after}")

    # Convert the "Date" column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort the DataFrame by date in ascending order
    df.sort_values(by="Date", inplace=True)
    print("Data sorted by date.")

    # Display the preprocessed DataFrame
    print("\nPreprocessed Data:")
    print(df.head())

    return df

# Replace "your_stock_data.csv" with the actual path to your .csv file
file_path = "C:\\Users\\YOURUSERNAME\\Desktop\\GOOGL_data.csv"
preprocessed_data = preprocess_stock_data(file_path)

# Specify the path for the new .csv file on the desktop
output_file_path = "C:\\Users\\YOURUSERNAME\\Desktop\\preprocessed_stock_data.csv"

# Save the preprocessed data to a new .csv file
preprocessed_data.to_csv(output_file_path, index=False)

print(f"\nPreprocessed data saved to {output_file_path}")
