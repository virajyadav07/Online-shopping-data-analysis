import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("kmeans_model.pkl")

# Load scaler
scaler = joblib.load("scaler.pkl")

print("Customer Segmentation Prediction")
print("--------------------------------")

# Take customer input
recency = float(input("Enter Recency: "))
frequency = float(input("Enter Frequency: "))
monetary = float(input("Enter Monetary: "))

# Create DataFrame
customer = pd.DataFrame({
    "Recency": [recency],
    "Frequency": [frequency],
    "Monetary": [monetary]
})
# Apply log transformation
customer_log = np.log1p(customer)
# Apply scaling
customer_scaled = scaler.transform(customer_log)
# Predict cluster
prediction = model.predict(customer_scaled)

print("\nPredicted Customer Cluster:", prediction[0])
