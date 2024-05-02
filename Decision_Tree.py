import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Reading the dataset
df = pd.read_csv("heart.csv")

# Feature scaling
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["chol", "age", "sex"]])

# Performing KMeans clustering
kmeans = KMeans(n_clusters=3)
df["cluster"] = kmeans.fit_predict(df_scaled)

# Splitting features and target variable
X = df_scaled
y = df["cluster"]

# Initialize DecisionTreeClassifier model
model = DecisionTreeClassifier(max_depth=3)

# Train the model
model.fit(X, y)

plot_tree(model, feature_names=["Cholesterol", "Age", "Sex"], class_names=[str(i) for i in range(3)], filled=True)
plt.show()
