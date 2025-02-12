#kopiert fra egen jupyter notebook

import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = "agaricus-lepiota.csv"
kolonner = ["klasse", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
           "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
           "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
           "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]

data = pd.read_csv(file_path, header=None, names=kolonner)

label_encoders = {}
for kol in data.columns:
    le = LabelEncoder()
    data[kol] = le.fit_transform(data[kol])
    label_encoders[kol] = le

data.head()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_verdier = list(range(2, 31))
silhouette_scores = []

for k in k_verdier:
    kmeans = KMeans(n_clusters=k, random_state=42) # oppretter kmeans-modell som tar inn k antall klynger
    labels = kmeans.fit_predict(data) # Bruker kmeans-modellen til å predikere klynger
    silhouette_avg = silhouette_score(data, labels) # Bruker silhouette_score til å evaluere klyngene.
    silhouette_scores.append(silhouette_avg) # Legger til silhouette_score i en liste

silhouette_scores

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(k_verdier, silhouette_scores, marker='o')
plt.title("Silhouette Score for K-means Clustering")
plt.xlabel("Antall klynger (k)")
plt.ylabel("Silhouette Score")
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

kmeans_optimal = KMeans(n_clusters=3, random_state=42)
labels_optimal = kmeans_optimal.fit_predict(data)

plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels_optimal, cmap='viridis')
plt.title("PCA-projeksjon av mushroom-datasettet")
plt.xlabel("Hovedkomponent 1")
plt.ylabel("Hovedkomponent 2")
plt.show()