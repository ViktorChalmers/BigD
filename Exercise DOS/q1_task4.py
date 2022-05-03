from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import rand_score, adjusted_mutual_info_score

def plot_cluster_ind():
    #iterate over different cluster sizes and internal clusters
    cluster_range = range(2,8+1)
    silhouette = np.zeros(len(cluster_range))
    davies_b = np.zeros(len(cluster_range))
    calinski_h = np.zeros(len(cluster_range))

    for j,i in enumerate(cluster_range):
        #kmeans = KMeans(n_clusters=i)
        #kmeans.fit(x_pca) 
        #silhouette[j] = silhouette_score(x_pca,kmeans.labels_)
        #davies_b[j] = davies_bouldin_score(x_pca,kmeans.labels_)
        #calinski_h[j] = calinski_harabasz_score(x_pca,kmeans.labels_)

        #gmm = GaussianMixture(n_components=i)
        #gmm.fit(x_pca) 
        #pred = gmm.fit_predict(X=x_pca)
        #silhouette[j] = silhouette_score(x_pca,pred)
        #davies_b[j] = davies_bouldin_score(x_pca,pred)
        #calinski_h[j] = calinski_harabasz_score(x_pca,pred)

        brc = Birch(threshold=0.01,branching_factor = 80,n_clusters=i)
        pred = brc.fit_predict(x_pca)
        silhouette[j] = silhouette_score(x_pca,pred)
        davies_b[j] = davies_bouldin_score(x_pca,pred)
        calinski_h[j] = calinski_harabasz_score(x_pca,pred)

    title = 'BIRCH'
    plt.plot(cluster_range,silhouette)
    plt.title(title)
    plt.xlabel('Cluster Count')
    plt.ylabel('Silhouette Score')
    plt.figure()
    plt.plot(cluster_range,davies_b)
    plt.title(title)
    plt.xlabel('Cluster Count')
    plt.ylabel('Davies-Bouldin Score')
    plt.figure()
    plt.plot(cluster_range,calinski_h)
    plt.title(title)
    plt.xlabel('Cluster Count')
    plt.ylabel('Calinski-Harabasz Score')
    plt.show()

def plot_true_labels():
    a = pd.read_csv('C:/Users/Jesper/OneDrive/Dokument/GitHub/BigD/Exercise DOS/labels.csv',index_col=0)
    a = a.to_numpy()
    a = a.flatten()

    x_pca['labels'] = a

    sns.pairplot(x_pca ,hue='labels',x_vars=[0,1,2,3,4],y_vars=[0,1,2,3,4])
    plt.show()

data = pd.read_csv('C:/Users/Jesper/OneDrive/Dokument/GitHub/BigD/Exercise DOS/data.csv',index_col=0,low_memory=False)

normalized = normalize(data)
data_scaled = pd.DataFrame(normalized)
variance = data_scaled.var()
columns = data.columns
meanVariance = variance.mean()

variable = []

for i in range(0,len(variance)):
    if variance[i] >=  meanVariance: #setting the threshold as mean variance
        variable.append(columns[i])

#print(f"Dropping {len(variance)-len(variable)} features < mean variance")

# creating a new dataframe using the above variables
new_data = data[variable]

#Normailizing the variance dropped data
drop_n_norm = normalize(new_data)


pca = PCA()
pca.fit(drop_n_norm)


chungus_len = 10 #Found to be good
pca = PCA(n_components=chungus_len)
pca.fit(drop_n_norm)
x_pca=pd.DataFrame(pca.transform(drop_n_norm))

#plots indiceses for different cluster counts
plot_cluster_ind()

#Plot true labels
#plot_true_labels()

a = pd.read_csv('C:/Users/Jesper/OneDrive/Dokument/GitHub/BigD/Exercise DOS/labels.csv',index_col=0)
a = a.to_numpy()
a = a.flatten()

brc = Birch(threshold=0.01,branching_factor = 80,n_clusters=5)
pred = brc.fit_predict(x_pca)
x_pca['labels'] = brc.fit_predict(x_pca)

#Pair plot
sns.pairplot(x_pca ,hue='labels',x_vars=[0,1,2,3,4],y_vars=[0,1,2,3,4])
plt.show()

print(f'Rand Index = {rand_score(a,pred)}')
print(f'Mutual Information Score = {adjusted_mutual_info_score(a,pred)}')
