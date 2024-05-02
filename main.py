import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

def KMeans_Algorithm(data):
    # Reading the dataset
    df = data.copy()

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
    st.pyplot()

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
    st.subheader("Cluster Visualization")
    plt.scatter(df1.chol, df1.age, df1.sex, color="blue", label="Cluster 1")
    plt.scatter(df2.chol, df2.age, df2.sex, color="red", label="Cluster 2")
    plt.scatter(df3.chol, df3.age, df3.sex, color="black", label="Cluster 3")
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color="purple", marker="*", label="Centroid")

    plt.xlabel("Cholesterol")
    plt.ylabel("Age")
    plt.legend()
    st.pyplot()

def DecisionTree_Algorithm(data):
    # Reading the dataset
    df = data.copy()

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
    st.pyplot()

def load_dataframe():
    # Load the dataset
    data = pd.read_csv("heart.csv")
    return data

def main():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Info", "Data Frame", "2D Visualization Tab", "K-Means Algorithm", "Decision Tree Algorithm", "Results"])
    data = load_dataframe()
    with tab1:
        pass

    with tab2:
        st.header("Heart Decease Data Frame")
        st.dataframe(data, width=800, height=600)
    
    with tab3:
        pass

    with tab4:
        st.header("K-Means Algorithm")
        KMeans_Algorithm(data)

    with tab5:
        st.header("Decision Tree Algorithm")
        DecisionTree_Algorithm(data)

    with tab6:
        pass
    
if __name__ == "__main__":
    main()