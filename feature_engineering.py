import pandas as pd
import numpy as np

# Load datasets
orders = pd.read_csv(r"C:\Users\malti\OneDrive\Documents\Data science project\olist_orders_dataset.csv\olist_orders_dataset.csv")

customers = pd.read_csv(r"C:\Users\malti\OneDrive\Documents\Data science project\olist_customers_dataset.csv\olist_customers_dataset.csv")

items = pd.read_csv(r"C:\Users\malti\OneDrive\Documents\Data science project\olist_order_items_dataset.csv\olist_order_items_dataset.csv")

payments = pd.read_csv(r"C:\Users\malti\OneDrive\Documents\Data science project\olist_order_payments_dataset.csv\olist_order_payments_dataset.csv")

reviews = pd.read_csv(r"C:\Users\malti\OneDrive\Documents\Data science project\olist_order_reviews_dataset.csv\olist_order_reviews_dataset.csv")

# Check data
print(orders.head())
print(orders.shape)
print(orders.info())

# Convert timestamp column into datetime format

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)

# # Extract Year
orders["purchase_year"] = (
    orders["order_purchase_timestamp"].dt.year
)

# # Extract Month
orders["purchase_month"] = (
    orders["order_purchase_timestamp"].dt.month
)

# # Extract Day
orders["purchase_day"] = (
    orders["order_purchase_timestamp"].dt.day
)

# # Extract Hour
orders["purchase_hour"] = (
    orders["order_purchase_timestamp"].dt.hour
)

# # Extract Weekday
orders["purchase_weekday"] = (
    orders["order_purchase_timestamp"].dt.day_name()
)

# # Check Weekend or Weekday
orders["is_weekend"] = (
    orders["order_purchase_timestamp"].dt.dayofweek >= 5
)

# Extract Quarter
orders["purchase_quarter"] = (
    orders["order_purchase_timestamp"].dt.quarter
)

# # Display result
print(orders.head())

# Weekend Feature Create karo
orders["is_weekend"] = (
    orders["order_purchase_timestamp"].dt.dayofweek >= 5
)

print(orders[[
    "order_purchase_timestamp",
    "is_weekend"
]].head())

# Delivery Delay Feature
orders["order_delivered_customer_date"] = pd.to_datetime(
    orders["order_delivered_customer_date"]
)

orders["delivery_days"] = (
    orders["order_delivered_customer_date"] -
    orders["order_purchase_timestamp"]
).dt.days

print(orders[[
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "delivery_days"
]].head())

orders["order_estimated_delivery_date"] = pd.to_datetime(
    orders["order_estimated_delivery_date"]
)

orders["delivery_delay_days"] = (
    orders["order_delivered_customer_date"] -
    orders["order_estimated_delivery_date"]
).dt.days

orders["delivered_late"] = (
    orders["delivery_delay_days"] > 0
)

# Total Order Value Feature
items["total_price"] = (
    items["price"] + items["freight_value"]
)

order_features = items.groupby("order_id").agg(
    total_items=("product_id", "count"),

    unique_products=("product_id", "nunique"),

    total_product_price=("price", "sum"),

    total_freight=("freight_value", "sum")
).reset_index()

order_features["total_order_value"] = (
    order_features["total_product_price"] +
    order_features["total_freight"]
)

# Payment Feature Engineering
print(order_features.head())

payment_features = payments.groupby("order_id").agg(

    total_payment=("payment_value", "sum"),

    max_installments=("payment_installments", "max"),

    payment_types=("payment_type", "nunique")

).reset_index()

print(payment_features.head())

# Customer Feature Engineering
customer_data = orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left"
)

customer_data = customer_data.sort_values(
    "order_purchase_timestamp"
)

customer_data["previous_orders"] = (
    customer_data.groupby("customer_unique_id").cumcount()
)

customer_data["is_first_order"] = (
    customer_data["previous_orders"] == 0
)

# Days Since Previous Order
customer_data["days_since_previous_order"] = (

    customer_data.groupby("customer_unique_id")[
        "order_purchase_timestamp"
    ].diff().dt.days

)

print(customer_data[[
    "customer_unique_id",
    "previous_orders",
    "days_since_previous_order"
]].head(10))

# Review Feature Engineering
review_features = reviews.groupby("order_id").agg(
    average_review_score=("review_score", "mean")
).reset_index()

review_features["review_category"] = pd.cut(
    review_features["average_review_score"],
    bins=[0, 2, 3, 5],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

print(review_features.head())

# Merge All Engineered Features
final_data = customer_data.merge(
    order_features,
    on="order_id",
    how="left"
)

final_data = final_data.merge(
    payment_features,
    on="order_id",
    how="left"
)

final_data = final_data.merge(
    review_features,
    on="order_id",
    how="left"
)

print(final_data.head())
print(final_data.shape)

import os

os.makedirs("data/processed", exist_ok=True)

import os

os.makedirs("data/processed", exist_ok=True)

#   Final Dataset Export
final_data.to_csv(
    "data/processed/olist_feature_engineered.csv",
    index=False
)

print("Feature Engineering Completed Successfully!")

