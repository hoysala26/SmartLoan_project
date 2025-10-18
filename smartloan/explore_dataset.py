import pandas as pd

# Load the dataset
df = pd.read_csv('farmer_data.csv')

# Preview the first 5 rows
print("🔹 First 5 rows:")
print(df.head())

# Summary statistics
print("\n🔹 Summary statistics:")
print(df.describe())

# Check for missing values
print("\n🔹 Missing values:")
print(df.isnull().sum())

# Value counts for key columns
print("\n🔹 Loan Status distribution:")
print(df['Loan_Status'].value_counts())

print("\n🔹 Region distribution:")
print(df['Region'].value_counts())

print("\n🔹 Crop Type distribution:")
print(df['Crop_Type'].value_counts())
