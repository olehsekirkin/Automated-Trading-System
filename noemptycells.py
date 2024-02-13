import pandas as pd
import os
from pathlib import Path

# Load your .csv file into a DataFrame
file_path = "C:\\Users\\YOURUSERNAME\\Desktop\\output_with_technical_indicators.csv"
df = pd.read_csv(file_path)

# Drop rows with missing data
df = df.dropna()

# Get the desktop path
desktop_path = str(Path.home() / "Desktop")

# Save the cleaned DataFrame to the desktop
cleaned_file_path = os.path.join(desktop_path, "cleaned_file.csv")
df.to_csv(cleaned_file_path, index=False)

print(f"Rows with missing data removed. Cleaned file saved to {cleaned_file_path}")
