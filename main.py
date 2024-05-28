import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, silhouette_score

def KMeans_Algorithm(data, K):
    df = data.copy()#Δημιουργία αντιγράφου του ΣΔ
    X = df.iloc[:, :-1]  #Διαγραφή της στήλης στόχου

    #Εκπαίδευση του αλγορίθμου
    km = KMeans(n_clusters=K, random_state=0)
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
    st.pyplot(plt)

    # Υπολογισμός της ακρίβειας του αλγορίθμου
    silhouette = silhouette_score(X, y_predicted)
    st.write(f"Silhouette Score: {silhouette*100:.2f}%")


def DecisionTree_Algorithm(data):
    df = data.copy() #Δημιουργία αντιγράφου του ΣΔ
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
    st.pyplot(plt)

    # Εκτύπωση του μήκους των X_train and y_train
    st.write(f"Length of X_train: {len(X_train)}")
    st.write(f"Length of X_test: {len(X_test)}")

    # Υπολογισμός της ακρίβειας του αλγορίθμου
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy Score: {accuracy*100:.2f}%")

def plot_data_pca(df):
    
    X = df.iloc[:, :-1]  #Χαρακτηριστικά του ΣΔ εκτός της τελευταία στήλης
    y = df.iloc[:, -1]  #Χαρακτηριστικά του ΣΔ την τελευταία στήλη 
    #Παρουσίαση δεδομένων με pca
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    
    plt.figure(figsize=(6, 3)) 
    scatter = plt.scatter(components[:, 0], components[:, 1], c=y, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Target')
    plt.title('2D PCA Visualization')
    st.pyplot(plt)

def plot_data_tsne(df):
    
    features = df.iloc[:, :-1]  #Χαρακτηριστικά του ΣΔ εκτός της τελευταία στήλης
    target = df.iloc[:, -1]  #Χαρακτηριστικά του ΣΔ την τελευταία στήλη στόχο

    #Παρουσίαση δεδομένων με tsne
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(features)

    fig = px.scatter(
        x=projections[:, 0], y=projections[:, 1],
        color=target.astype(str), labels={'color': target.name}
    )
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig)

def plot_histograms(df):
    #Παρουσίαση δεδομένων με ιστόγραμμα
    st.header("Histograms")
    for column in df.columns[0:2]:
        plt.figure(figsize=(6, 3)) 
        plt.hist(df[column], bins=30, edgecolor='k')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(plt)

def plot_pair_plots(df):
    #Παρουσίαση δεδομένων με pairplots
    st.header("Pair Plots")
    sns.pairplot(df, hue=df.columns[-1]) 
    st.pyplot(plt)

def main():
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Info", "Data Frame", "2D Visualization Tab", "K-Means Algorithm", "Decision Tree Algorithm", "Results"])

    data = None 

    with tab1:
        st.header("Πληροφρίες:")
        st.write("---------------")
        st.write("""Αυτή η εφαρμογή επιτρέπει στο χρήστη, να ανεβάσει ένα σύνολο δεδομένων και στην συνέχεια,
                 να προχωρίσει στην ανάλυση τους με την βοήθεια μηχανικής μάθησης. Ποιο συγκεκριμένα, μπορεί να
                 αναλύσει τα δεδομένα με την βοήθεια των αλγορίθμων K-means και Decision tree. Επίσης, τα δεδομένα
                 παρουσιάζονται σε  γραφήματα μέσω ιστογράμματα, scatter plots, TsNe plots και PCA plots.
                """)
        st.header("Χρήση:")
        st.write("---------------")
        st.write("""Μέσω της καρτέλας 'Data Frame' ο χρήστης μπορεί να ανεβάση το σύνολο δεδομένων του και να μελετήσει 
                 τα διαγράμματα που παρουσιάζονται με βάση το σύνολο δεδομένων. Στην συνέχεια μπορεί να αναλύσει τα δεδομένα με τον
                 αλγόριθμο K-means μέσω της καρτέλας 'K-means algorithm', ή με τον αλγόριθμο Decision tree μέσω της καρτέλας 
                 'Decision tree algorithm'. Tέλος, μέσω της καρτέλας 'Results',μπορεί να διαβάσει τις συγκρίσεις και τα αποτελέσματα
                 των δύο αλγορίθμων για το σύνολο δεδομένων που χρησιμποιήσαμε εμείς.
                """)

    with tab2:
        st.header("Σύνολο δεδομένων")
        uploaded_file = st.file_uploader("Choose a file")#File uploader

        if uploaded_file is not None:#Ελεγχος του αρχείου που αναιβάζει ο χρήστης για το αν είναι κενό

            if uploaded_file.name.endswith('.csv'):#Ελεγχος για τον τύπο αρχείου 
                #Ανάγνωση και παρουσίαση του συνόλου δεδομένων
                data = pd.read_csv(uploaded_file)
                st.dataframe(data, width=800, height=600)
                st.header("Exploratory Data Analysis (EDA)")
                plot_histograms(data)
                plot_pair_plots(data)
            elif uploaded_file.name.endswith('.xlsx'):#Ελεγχος για τον τύπο αρχείου
                data = pd.read_excel(uploaded_file)
                st.dataframe(data, width=800, height=600)
                st.header("Exploratory Data Analysis (EDA)")
                plot_histograms(data)
                plot_pair_plots(data)
            else:
                st.error("Invalid file format. Please upload a .csv or .xlsx file.")
        

        with tab3:
            #Παρουσίαση των δεδομένων με pca και tsne
            if data is not None:
                st.header("2D Visualizations")
                plot_data_tsne(data)
                plot_data_pca(data)
            else:
                st.header("Δεν βρέθηκε σύνολο δεδομένων.")
                st.write("Δοκιμάστε να ανεβάσετε το σύνολο δεδομένων σα μέσω της καρτέλας 'Data Frame'.")

        with tab4:
            #Εκτέλεση του αλγορίθμου k-means
            if data is not None:
                st.header("Αλγόριθμος K-Means")
                K = st.number_input("Number of clusters", min_value=1, max_value=10, value=3, step=1)
                if st.button("Submit"):
                    KMeans_Algorithm(data, int(K))
            else:
                st.header("Δεν βρέθηκε σύνολο δεδομένων.")
                st.write("Δοκιμάστε να ανεβάσετε το σύνολο δεδομένων σα μέσω της καρτέλας 'Data Frame'.")

        with tab5:
            #Εκτέλεση του αλγορίθμου Decision Tree
            if data is not None:
                st.header("Αλγόριθμος Decision Tree ")
                DecisionTree_Algorithm(data)
            else:
                st.header("Δεν βρέθηκε σύνολο δεδομένων.")
                st.write("Δοκιμάστε να ανεβάσετε το σύνολο δεδομένων σα μέσω της καρτέλας 'Data Frame'.")
        
        with tab6:
            st.header("Σύγκριση:")
            st.write("---------------")
            st.subheader("Πως λειτουργεί ο αλγόριθμος Decision tree;")
            st.write(""" 
                    Ο αλγόριθμος decision tree αρχικά δημιουργεί ένα μοντέλο που προβλέπει την τιμή στόχο με βάση τους κανόνες 
                    από τα χαρακτηριστικά εισόδου του συνόλου δεδομένων.Στην συνέχεια, χωρίζει το σύνολο δεδομένων σε μικρότερα 
                    υποσύνολα με βάση τα χαρακτηριστικά, δημιουργώντας ένα δέντρο αποφάσεων.Σε κάθε κόμβο του δέντρου λαμβάνει 
                    απόφαση για το πώς να χωρίσει τα δεδομένα χρησιμοποιώντας κριτήρια όπως η εντροπία ή το Gini index.
                    \nΧρησιμος για:
                        \n(1)κατηγοριοποίηση και πρόβλεψη με δεδομένα με ετικέτες,
                        \n(2)Kατανόηση των σχέσεων μεταξύ των μεταβλητών.
                    """)
            st.subheader("Πως λειτουργεί ο αλγόριθμος K-means;")
            st.write(""" 
                    Αντίθετα, ο αλγόριμός K-means, χωρίζει τα δεδομένα σε K ομάδες, όπου κάθε ομάδα έχει ένα κέντρο.
                    Ο K-means βασίζεται σε μετρήσεις αποστάσεων, συνήθως την ευκλείδεια απόσταση, για να αντιστοιχίσει κάθε δείγμα από το σύνολο δεδομένων στο πλησιέστερο κέντρο.
                    Έτσι, σχηματίζονται οι ομάδες των δεδομένων με βάση τα κοινά χαρακτηριστικά τους.\\
                    Χρησιμος για:\n
                    \nΧρησιμος για:
                        \n(1)Ομαδοποίηση χωρίς ετικέτες,
                        \n(2)Ανακάλυψη ομοιοτήτων και μοτίβων.
                    """)
            st.header("Αποτελέσματα:")
            st.write("---------------")
            st.write("""
                    Για το σύνολο δεδομένων που χρησιμοποιήσαμε, ο αλγόριθμος K-means είχε ποσοστό επιτυχίας
                    περίπου 28\% ενώ ο αλγόριθμος Decision tree είχε ποσοστό επιτυχίας περίπου 90\%. Για την εκπαίδευση 
                    των μοντέλων, χρησιμοποιήθηκαν 242 δείγματα και για τον έλεγχο 61 δείγματα.
                    Με βάση τα αποτελέσματά, ο αλγόριθμος  Decision tree είναι ακριβέστερος για την ανάλυση των δεδομένων.
                    Ο λόγος που είναι ποιο αποτελεσματικός, είναι γιατί το σύνολο δεδομένων μας περιείχε δεδομένα από άτομα
                    που έπασχαν από καρδιακά προβλήματά και από μη πάσχων άτομα. Έτσι, ο αλγόριθμος, με την βοήθεια των ετικετών,
                    μπόρεσε να κατανόησει τα μοτίβα με τα χαρακτηριστικά των ασθενών και να καταλήξει σε συμπέρασμα για το αν κάποιος
                    πάσχει από καρδιακά προβλήματα ή όχι.
                    """)
if __name__ == "__main__":
    main()
