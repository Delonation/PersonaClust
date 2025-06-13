# Confidential Simplified Clustering Analysis for PersonaClust
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Load anonymized respondent data
data = pd.read_csv("respondent_responses.csv")

# Feature extraction (complex logic simplified)
features = data[['score1', 'score2', 'score3', 'score4']]
features.fillna(features.mean(), inplace=True)

# Apply k-means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
data['cluster_label'] = kmeans.fit_predict(features)

# Cluster visualization (scatter plot with centroids)
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue', 'purple']
for i in range(4):
    plt.scatter(features.values[data['cluster_label'] == i, 0],
                features.values[data['cluster_label'] == i, 1],
                color=colors[i], label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=250, c='yellow', edgecolor='black', marker='X')

plt.title('K-Means Clustering of Respondent Traits', fontsize=15)
plt.xlabel('Trait Score 1')
plt.ylabel('Trait Score 2')
plt.legend()
plt.grid(True)

plt.savefig('clusters_analysis.png')
plt.close()
