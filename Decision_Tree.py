import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

#Διάβασμα του dataset
df = pd.read_csv("heart.csv")

X = df.iloc[:, :-1]  #Χαρακτηριστικά
y = df.iloc[:, -1]   #Στήλη στόχος

#Διαχωρισμός δεδομένων σε train και test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Κλιμάκωση των χαρακτηριστικών
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Εκπαίδευση του μοντέλου
model = DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(X_train_scaled, y_train)

# Παρουσίαση του decision tree
plt.figure(figsize=(10, 5))  
plot_tree(model, feature_names=df.columns[:-1], class_names=[str(i) for i in set(y)], filled=True)
plt.show()

# Εκτύπωση του μήκους των X_train and y_train
print(f"Length of X_train: {len(X_train)}")
print(f"Length of X_test: {len(X_test)}")

# Υπολογισμός της ακρίβειας του αλγορίθμου
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy*100:.2f}%")