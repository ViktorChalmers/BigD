from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import rand_score, adjusted_mutual_info_score

#from sklearn.metrics import davies_bouldin_score
#from jqmcvi import base #Dunn index

def optimise_k_means(data, max_k):
    #Elbow plot, plotting the inertia. Which is a measure of how well the data was clustered by k-means
    #lopar igenom flera olika klusterstorlekar
    #identifying optimum number of clusters
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        means.append(k)
        inertias.append(kmeans.inertia_)

    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()

data = pd.read_csv('C:/Users/Jesper/OneDrive/Dokument/GitHub/BigD/Exercise DOS/data.csv',index_col=0,low_memory=False)
 
'''
print(f"{data.head()} Head of data \n {data.shape[0]} datapoints, {data.shape[1]} features")
print(f"{data.isnull().sum() / len(data) * 100}, -> we dont have any missing values in the data")
print(data.dtypes)
print("normalizing data")
'''

normalized = normalize(data)
data_scaled = pd.DataFrame(normalized)
variance = data_scaled.var()
columns = data.columns
#print(variance.head())
meanVariance = variance.mean()

variable = []

for i in range(0,len(variance)):
    if variance[i] >=  meanVariance: #setting the threshold as mean variance
        variable.append(columns[i])

print(len(variance))
print(len(variable))
print(f"Dropping {len(variance)-len(variable)} features < mean variance")

# creating a new dataframe using the above variables
new_data = data[variable]

'''
# first five rows of the new data
print(new_data.head())
print(f"New data size: {new_data.shape}, old data size: {data.shape}")
#variance of variables in new data
print(new_data.head().var())
plt.plot(variance,  "*")
plt.ylabel('Variance')
plt.xlabel('Feature')
plt.axhline(meanVariance,   color="red")
plt.figure()
plt.axvline(meanVariance,   color="red")
plt.hist(variance,bins = 50)
plt.xlabel('Variance')
plt.ylabel('Frequency')
plt.show()
'''

'''
mean_data = new_data.mean()
var_data = new_data.var()

plt.hist(mean_data,bins = 50)
plt.xlabel('Mean')
plt.figure()
plt.hist(var_data,bins = 50)
plt.xlabel('Variance')
plt.show()
'''

drop_n_norm = normalize(new_data)

pca = PCA()
pca.fit(drop_n_norm)

#x_pca=pca.transform(drop_n_norm)
#print(f'Singular values ({len(pca.singular_values_)}): \n{pca.singular_values_}')
#print(f"Reducing dimensions from {drop_n_norm.shape} to {x_pca.shape}")

#cut_off = 0.025
cut_off = 0.01

'''
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2)
#plt.plot(PC_values, pca.explained_variance_, 'o-', linewidth=2)
plt.axhline(cut_off,   color="red")
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained Ratio')
plt.show()
'''


chungus_len = len(pca.explained_variance_ratio_[pca.explained_variance_ratio_ > cut_off])
print(chungus_len) #10 gives many nice results

pca = PCA(n_components=chungus_len)
#pca = PCA(n_components=5)
pca.fit(drop_n_norm)
x_pca=pd.DataFrame(pca.transform(drop_n_norm))

'''
PC_values = np.arange(pca.n_components_) + 1
#plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()
'''

#sns.pairplot(x_pca)
#plt.show()

#max_k = 15
#optimise_k_means(x_pca, max_k)

a = pd.read_csv('C:/Users/Jesper/OneDrive/Dokument/GitHub/BigD/Exercise DOS/labels.csv',index_col=0)
a = a.to_numpy()
a = a.flatten()
#5 clusters good
kmeans = KMeans(n_clusters=5)
kmeans.fit(x_pca) 
x_pca['labels'] = kmeans.labels_
pred = kmeans.labels_

#gmm = GaussianMixture(n_components=5)
#gmm.fit(x_pca) 
#x_pca['labels'] = gmm.fit_predict(X=x_pca)

print(f'Rand Index = {rand_score(a,pred)}')
print(f'Mutual Information Score = {adjusted_mutual_info_score(a,pred)}')

#sns.pairplot(x_pca ,hue='labels',x_vars=[0,1,2,3,4],y_vars=[0,1,2,3,4])
#plt.show()

#2 do: Compare with true label
#sklearn.metrics.homogeneity_score compare with true data

