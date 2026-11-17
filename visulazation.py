import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    r"C:\Users\malti\OneDrive\Documents\Data science project\Copy of Online Retail.csv",
    encoding="cp1252"
)

# print("Dataset Loaded Successfully!")
# print(df.head())


# ==========================================
# 2. DATA PREPROCESSING
# ==========================================

# Convert InvoiceDate into datetime
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

# Create Revenue column
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Remove cancelled invoices and invalid sales rows
sales = df[
    (~df["InvoiceNo"].astype(str).str.startswith("C")) &
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0) &
    (df["InvoiceDate"].notna())
].copy()


# Extract date features
sales["Year"] = sales["InvoiceDate"].dt.year
sales["Month"] = sales["InvoiceDate"].dt.month
sales["Hour"] = sales["InvoiceDate"].dt.hour
sales["Weekday"] = sales["InvoiceDate"].dt.day_name()

print("\nSales Dataset Shape:", sales.shape)


# ==========================================
# 3. MONTHLY REVENUE VISUALIZATION
# ==========================================

monthly_revenue = (
    sales.groupby(sales["InvoiceDate"].dt.to_period("M"))["Revenue"]
    .sum()
)

plt.figure(figsize=(12, 6))
monthly_revenue.plot(kind="line", marker="o")

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (£)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 4. TOP 10 PRODUCTS VISUALIZATION
# ==========================================

top_products = (
    sales.groupby("Description")["Quantity"]
    .sum()
    .nlargest(10)
    .sort_values()
)

plt.figure(figsize=(12, 6))
top_products.plot(kind="barh")

plt.title("Top 10 Best Selling Products")
plt.xlabel("Quantity Sold")
plt.ylabel("Product Name")
plt.tight_layout()
plt.show()


# ==========================================
# 5. COUNTRY-WISE REVENUE
# ==========================================

country_revenue = (
    sales.groupby("Country")["Revenue"]
    .sum()
    .nlargest(10)
    .sort_values()
)

plt.figure(figsize=(12, 6))
country_revenue.plot(kind="barh")

plt.title("Top 10 Countries by Revenue")
plt.xlabel("Revenue (£)")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# ==========================================
# 6. UNIT PRICE DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=sales,
    x="UnitPrice",
    bins=50
)

plt.xlim(0, sales["UnitPrice"].quantile(0.99))
plt.title("Unit Price Distribution")
plt.xlabel("Unit Price (£)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ==========================================
# 7. SCATTER PLOT
# ==========================================

sample_data = sales.sample(
    n=min(5000, len(sales)),
    random_state=42
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sample_data,
    x="UnitPrice",
    y="Quantity"
)

plt.title("Quantity vs Unit Price")
plt.xlabel("Unit Price (£)")
plt.ylabel("Quantity")
plt.tight_layout()
plt.show()

# ==========================================
# 8. SAVE VISUALIZATION DATA
# ==========================================

sales.to_csv(
    "Online_Retail_Visualization_Data.csv",
    index=False
)

print("\nAll Visualizations Completed Successfully!")

