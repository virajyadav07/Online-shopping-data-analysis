import pandas as pd
import numpy as np
import joblib
pip install -q streamlit pandas numpy matplotlib

# App.py
%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Online Retail Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Online Retail Data Analysis")
st.write("Sales Performance and Customer Analytics Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader(
    "Upload Online Retail Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, encoding="cp1252")

    # Data cleaning
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=["CustomerID", "InvoiceDate"]
    ).copy()

    df = df[
        ~df["InvoiceNo"].astype(str).str.startswith("C")
    ].copy()

    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ].copy()

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # KPI calculations
    total_revenue = df["Revenue"].sum()
    total_orders = df["InvoiceNo"].nunique()
    total_customers = df["CustomerID"].nunique()
    total_quantity = df["Quantity"].sum()

    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Revenue", f"£{total_revenue:,.2f}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Total Customers", f"{total_customers:,}")
    col4.metric("Total Quantity", f"{total_quantity:,}")

    st.divider()

    # Monthly revenue
    st.subheader("Monthly Revenue Trend")

    monthly_revenue = (
        df.set_index("InvoiceDate")
        .resample("ME")["Revenue"]
        .sum()
    )

    st.line_chart(monthly_revenue)

    # Top products
    st.subheader("Top 10 Products")

    top_products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    st.bar_chart(top_products)

    # Country revenue
    st.subheader("Country-wise Revenue")

  # Colab token set
!pip install pyngrok streamlit -q

from pyngrok import ngrok

ngrok.set_auth_token("3MBqenzGm66EhqpRSSsqmTATPdo_6QzxaxaGbHh5tDRa4n1br")


    country_revenue = (
        df.groupby("Country")["Revenue"]
        .sum()
        .nlargest(10)
    )

    st.bar_chart(country_revenue)

    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

else:
    st.info("Please upload your Online Retail CSV file to view the dashboard.")
  # Colab token set
!pip install pyngrok streamlit -q

from pyngrok import ngrok

ngrok.set_auth_token("3MBqenzGm66EhqpRSSsqmTATPdo_6QzxaxaGbHh5tDRa4n1br")

# Public URL generate
public_url = ngrok.connect(8501)
print("Your Dashboard URL:", public_url)
Your Dashboard URL: NgrokTunnel: "https://vindicate-expiring-doorbell.ngrok-free.dev" -> "http://localhost:8501"
