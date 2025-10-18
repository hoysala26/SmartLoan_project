import pandas as pd
import joblib
from fairlearn.metrics import MetricFrame, selection_rate, accuracy_score, false_positive_rate, true_positive_rate
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("model.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# Load original test set with sensitive features
df = pd.read_csv("farmer_data.csv")
df_encoded = df.copy()
df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 1, 'Female': 0})
df_encoded['Subsidy_Access'] = df_encoded['Subsidy_Access'].map({'Yes': 1, 'No': 0})
df_encoded['Previous_Loan'] = df_encoded['Previous_Loan'].map({'Yes': 1, 'No': 0})
df_encoded['Loan_Status'] = df_encoded['Loan_Status'].map({'Approved': 1, 'Rejected': 0})

# Match test set rows
test_indices = df_encoded.sample(frac=0.2, random_state=42).index
sensitive_features = df_encoded.loc[test_indices, ['Gender', 'Region', 'Subsidy_Access']]

# Predict
y_pred = model.predict(X_test)

# Fairness metrics
metrics = {
    "accuracy": accuracy_score,
    "selection_rate": selection_rate,
    "true_positive_rate": true_positive_rate,
    "false_positive_rate": false_positive_rate
}

# Analyze fairness across Gender
mf_gender = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features["Gender"])
print("\n🔍 Fairness by Gender:")
print(mf_gender.by_group)

# Plot
mf_gender.by_group.plot(kind="bar", figsize=(10,6))
plt.title("Fairness Metrics by Gender")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("fairness_gender.png")
print("✅ Fairness plot saved as fairness_gender.png")
