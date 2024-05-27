from sklearn.decomposition import PCA
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import matplotlib.pyplot as plt

def plot_data_pca(df):
    
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]  
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    
    plt.figure(figsize=(6, 3)) 
    scatter = plt.scatter(components[:, 0], components[:, 1], c=y, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Target')
    plt.title('2D PCA Visualization')
    plt.show()

def plot_data_tsne(df):
    
    features = df.iloc[:, :-1]  
    target = df.iloc[:, -1] 
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(features)

    fig = px.scatter(
        x=projections[:, 0], y=projections[:, 1],
        color=target.astype(str), labels={'color': target.name}
    )
    fig.update_traces(marker=dict(size=5))
    fig.show()

def main():
    df = pd.read_csv("heart.csv")
    plot_data_tsne(df)
    plot_data_pca(df)

if __name__ == "__main__":
    main()