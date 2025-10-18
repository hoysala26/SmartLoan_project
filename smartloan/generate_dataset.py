import pandas as pd
import random

# Sample values
genders = ['Male', 'Female']
crop_types = ['Wheat', 'Rice', 'Sugarcane', 'Maize', 'Cotton']
regions = ['Karnataka', 'Punjab', 'Bihar', 'Maharashtra', 'Tamil Nadu']
subsidies = ['Yes', 'No']

# Generate synthetic data
data = []
for i in range(100):
    age = random.randint(25, 65)
    gender = random.choice(genders)
    land_size = round(random.uniform(1.0, 10.0), 2)
    crop = random.choice(crop_types)
    income = random.randint(50000, 300000)
    credit_score = random.randint(300, 850)
    prev_loan = random.choice(['Yes', 'No'])
    current_debt = random.randint(0, 100000)
    region = random.choice(regions)
    subsidy = random.choice(subsidies)
    dependents = random.randint(0, 5)
    loan_status = random.choice(['Approved', 'Rejected'])

    data.append([
        age, gender, land_size, crop, income, credit_score,
        prev_loan, current_debt, region, subsidy, dependents, loan_status
    ])

# Create DataFrame
columns = [
    'Age', 'Gender', 'Land_Size', 'Crop_Type', 'Annual_Income',
    'Credit_Score', 'Previous_Loan', 'Current_Debt', 'Region',
    'Subsidy_Access', 'Dependents', 'Loan_Status'
]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('farmer_data.csv', index=False)
print("✅ farmer_data.csv generated successfully!")
