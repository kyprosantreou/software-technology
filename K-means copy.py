import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Reading the dataset
df = pd.read_csv("heart.csv")

# Elbow method for finding optimal k
k_rng = range(1, 10)
sse = []

# Calculating Sum of Squared Errors for different values of k
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[["chol","age","sex"]])
    sse.append(km.inertia_)

# Plotting Elbow curve
plt.xlabel("K")
plt.ylabel("Sum of Squared Error")
plt.title("Elbow Method")
plt.grid(True)
plt.plot(k_rng, sse)
plt.show()

#Applying KMeans clustering
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[["chol","age","sex"]])

# Adding cluster labels to the DataFrame
df["cluster"] = y_predicted

# Creating DataFrames for each cluster
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plotting clusters and centroids
plt.scatter(df1.chol, df1.age, df1.sex, color="blue", label="Cluster 1")
plt.scatter(df2.chol, df2.age, df2.sex, color="red", label="Cluster 2")
plt.scatter(df3.chol, df3.age, df3.sex, color="black", label="Cluster 3")
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color="purple", marker="*", label="Centroid")

plt.xlabel("Cholesterol")
plt.ylabel("Age")
plt.legend()
plt.show()