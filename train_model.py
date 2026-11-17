# Step 1: Libraries import
import numpy as np
import pandas as pd
import sklearn
import joblib

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Joblib:", joblib.__version__)

print("All libraries working!")
# Step 2:Check libraries working or not
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("All libraries imported successfully!")

# Step 3: CSV dataset upload karo

from google.colab import files

uploaded = files.upload()

file_name = next(iter(uploaded))

df = pd.read_csv(file_name, encoding="cp1252")

print("Dataset loaded successfully!")
print(df.head())

# Convert InvoiceDate into datetime
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    dayfirst=True,
    errors="coerce"
)

# Remove missing values
df = df.dropna(
    subset=["CustomerID", "InvoiceDate"]
).copy()

# Revenue column
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

print(df[["Quantity", "UnitPrice", "Revenue"]].head())

# Step 4: Create RFM features
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "Revenue": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

print(rfm.head())

# Step 5: Prepare data for Machine Learning
rfm_log = np.log1p(rfm)

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm_log)

print("Data scaling completed!")

# Step 6: Find optimal number of clusters
inertia = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(rfm_scaled)

    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(range(2, 11), inertia, marker="o")

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()

# Step 7: Train K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

print("Model training completed!")

score = silhouette_score(
    rfm_scaled,
    rfm["Cluster"],
    sample_size=min(2000, len(rfm)),
    random_state=42
)

print("Silhouette Score:", round(score, 3))

# Step 8: Analyze customer segments
cluster_analysis = rfm.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "Cluster": "count"
})

cluster_analysis = cluster_analysis.rename(
    columns={"Cluster": "CustomerCount"}
)

print(cluster_analysis.round(2))

# Step 9: Save trained model
joblib.dump(kmeans, "kmeans_model.pkl")

joblib.dump(scaler, "scaler.pkl")

rfm.to_csv("customer_segments.csv")

print("Model and scaler saved successfully!")


