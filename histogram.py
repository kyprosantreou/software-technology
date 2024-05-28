import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")

for column in df.columns[0:2]:
    plt.figure(figsize=(6, 3)) 
    plt.hist(df[column], bins=30, edgecolor='k')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
