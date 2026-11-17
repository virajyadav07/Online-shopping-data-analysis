
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================
# 1. LOAD ALL DATASETS
# =========================================
df = pd.read_csv(
    "Copy of Online Retail.csv",
    encoding="cp1252"
)



print("Dataset Loaded Successfully!")


# ==========================================
# 2. BASIC INFORMATION
# ==========================================
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# 3. MISSING VALUE ANALYSIS
# ==========================================
print("\nMissing Values:")
print(df.isnull().sum())

print("\nMissing Value Percentage:")
print((df.isnull().mean() * 100).round(2))

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.show()

# ==========================================
# 4. DUPLICATE ANALYSIS
# ==========================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ==========================================
# 5. DATA TYPE CONVERSION
# ==========================================

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

df["CustomerID"] = df["CustomerID"].astype("Int64")

# =========================================
# 6. FINAL MESSAGE
# =========================================

print("\nEDA ANALYSIS COMPLETED SUCCESSFULLY!")

