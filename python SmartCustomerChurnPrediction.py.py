import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("======================================")
print(" SMART CUSTOMER CHURN PREDICTION SYSTEM ")
print("======================================")

# Sample dataset
data = {
    "tenure": [12, 24, 5, 8, 36, 48, 2, 15, 20, 30, 10, 18],
    "monthly_charges": [50, 70, 90, 85, 60, 55, 95, 65, 72, 58, 88, 67],
    "support_calls": [1, 0, 5, 4, 0, 1, 6, 2, 3, 0, 5, 1],
    "churn": [0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

print("\nDataset Preview:")
print(df)

# Features and target
X = df[["tenure", "monthly_charges", "support_calls"]]
y = df["churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Prediction system
print("\n===== CUSTOMER PREDICTION =====")

tenure = int(input("Enter Customer Tenure (months): "))
monthly_charges = float(input("Enter Monthly Charges: "))
support_calls = int(input("Enter Number of Support Calls: "))

new_customer = np.array([[tenure, monthly_charges, support_calls]])
result = model.predict(new_customer)

print("\nPrediction Result:")
if result[0] == 1:
    print("Customer is likely to CHURN.")
else:
    print("Customer will STAY.")

print("\nProject Completed Successfully!")