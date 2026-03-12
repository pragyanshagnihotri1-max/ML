# Import libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ----------------------------
# Example Dataset: Customer Data
# ----------------------------
data = {
    'CustomerID': [1,2,3,4,5,6,7,8,9,10],
    'Annual_Income': [15,16,17,18,22,23,24,25,30,31],  # in $1000
    'Spending_Score': [39,81,6,77,40,76,6,94,45,67]    # Score out of 100
}

df = pd.DataFrame(data)

# Features for clustering
X = df[['Annual_Income','Spending_Score']]

# ----------------------------
# Apply KMeans Clustering
# ----------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Add cluster labels to DataFrame
df['Cluster'] = kmeans.labels_

print("Clustered Data:")
print(df)

# ----------------------------
# Visualize Clusters
# ----------------------------
plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'], cmap='rainbow')
plt.xlabel('Annual Income ($1000)')
plt.ylabel('Spending Score (0-100)')
plt.title('Customer Segmentation (K-Means Clustering)')
plt.show()
