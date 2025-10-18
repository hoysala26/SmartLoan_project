import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("model.pkl")
X_train = pd.read_csv("X_train.csv")

# Create a masker for tabular data
masker = shap.maskers.Independent(X_train)

# Use SHAP's general Explainer
explainer = shap.Explainer(model.predict, masker)
shap_values = explainer(X_train)

# Plot summary
shap.summary_plot(shap_values, X_train, show=False)
plt.savefig("shap_visuals.png")
print("✅ SHAP summary plot saved as shap_visuals.png")
