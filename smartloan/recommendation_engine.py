def generate_recommendations(applicant):
    tips = []

    if applicant['Credit_Score'] < 600:
        tips.append("Improve your credit score above 600 by repaying debts on time.")

    if applicant['Annual_Income'] < 100000:
        tips.append("Increase your annual income through crop diversification or government schemes.")

    if applicant['Current_Debt'] > 50000:
        tips.append("Reduce your current debt below ₹50,000 to improve eligibility.")

    if applicant['Land_Size'] < 2.0:
        tips.append("Consider leasing additional land or joining cooperative farming.")

    if applicant['Subsidy_Access'] == 'No':
        tips.append("Apply for agricultural subsidies to strengthen your financial profile.")

    if not tips:
        tips.append("Your profile is close to approval. Try reapplying with updated financials.")

    return tips

# Example usage
sample_applicant = {
    'Credit_Score': 580,
    'Annual_Income': 95000,
    'Current_Debt': 60000,
    'Land_Size': 1.5,
    'Subsidy_Access': 'No'
}

recommendations = generate_recommendations(sample_applicant)
print("\n🔍 Recommendations for Rejected Applicant:")
for tip in recommendations:
    print("-", tip)
