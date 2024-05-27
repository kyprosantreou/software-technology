import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

df = pd.read_csv("heart.csv")
#Διαγραφή της τελευταίας
X = df.iloc[:, :-1]  

#Εκπαίδευση του αλγορίθμου k-means
km = KMeans(n_clusters=5, random_state=0)
y_predicted = km.fit_predict(X)

df["cluster"] = y_predicted

#Παρουσίαση των clusters
plt.figure(figsize=(6, 3))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df["cluster"], cmap='viridis')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color="red", marker="x", s=100, label="Centroids")
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title("K-Means Clustering")
plt.legend()
plt.show()

#Υπολογισμός του score
silhouette = silhouette_score(X, y_predicted)
print(f"Silhouette Score: {silhouette*100:.2f}%")