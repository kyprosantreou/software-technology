import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart.csv")

sns.pairplot(df, hue=df.columns[-1]) 
plt.show()
