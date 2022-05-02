from matplotlib.colors import Normalize, LogNorm
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def optimise_GMM(data):
    # Checks for optimal number of components in GMM by plotting BIC vs AIC
    n_components = np.arange(1, 21)
    models = [GaussianMixture(n_components=n).fit(data) for n in n_components]

    plt.plot(n_components, [model.bic(data) for model in models], label='BIC')
    plt.plot(n_components, [model.aic(data) for model in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.show()

def optimise_k_means(data, max_k):
    # Elbow plot, plotting the inertia. Which is a measure of how well the data was clustered by k-means
    # lopar igenom flera olika klusterstorlekar
    # identifying optimum number of clusters
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


def filterData():
    from sklearn.preprocessing import normalize
    try:
        data = pd.read_pickle("data.pkl")
    except:
        try:  # load data if pkl exists
            data = pd.read_pickle("data.pkl")
        except:
            try:  # loading data from csv if extracted and save in pkl
                data = pd.read_csv("TCGA-PANCAN-HiSeq-801x20531/data.csv", index_col=0)
                data.to_pickle("data.pkl")
            except:  # extracting reading and saving data in pkl
                import tarfile

                file = tarfile.open('TCGA-PANCAN-HiSeq-801x20531.tar.gz')
                file.extractall()

                data = pd.read_csv("TCGA-PANCAN-HiSeq-801x20531/data.csv", index_col=0)
                data.to_pickle("data.pkl")

    normalize = normalize(data)
    data_scaled = pd.DataFrame(normalize)
    variance = data_scaled.var()
    columns = data.columns

    meanVariance = variance.mean()

    variable = []

    for i in range(0, len(variance)):
        if variance[i] >= meanVariance:  # setting the threshold as mean variance
            variable.append(columns[i])

    print(f"Dropping {len(variance) - len(variable)} features < mean variance")

    # creating a new dataframe using the above variables
    new_data = data[variable]
    plt.plot(variance, "*")
    plt.axhline(meanVariance, color="red")
    plt.figure()
    plt.axvline(meanVariance, color="red")
    plt.hist(variance, bins=50)
    # plt.show()
    new_data.to_pickle("new_data.pkl")
    return new_data

try:
    new_data = pd.read_pickle("new_data.pkl")
except:
    new_data = filterData()




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

# x_pca=pca.transform(drop_n_norm)
# print(f'Singular values ({len(pca.singular_values_)}): \n{pca.singular_values_}')
# print(f"Reducing dimensions from {drop_n_norm.shape} to {x_pca.shape}")

cut_off = 0.02

'''
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2)
plt.axhline(cut_off,   color="red")
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()
'''

chungus_len = len(pca.explained_variance_ratio_[pca.explained_variance_ratio_ > cut_off])
print(chungus_len)  # 10

pca = PCA(n_components=chungus_len)
pca.fit(drop_n_norm)
x_pca = pd.DataFrame(pca.transform(drop_n_norm))

'''
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()
'''

# sns.pairplot(x_pca)
# plt.show()

#max_k = 15
# optimise_k_means(x_pca, max_k)

# 5 clusters good
kmeans = KMeans(n_clusters=5)
kmeans.fit(x_pca)  # or new data??

optimise_GMM(x_pca)
GMM = GaussianMixture(n_components=5)
# plt.scatter(x_pca.loc[:,0],x_pca.loc[:,1],c=kmeans.labels_)

# hue??, palette

x_pca['labels'] = kmeans.labels_
#x_pca['labels'] = GMM.fit_predict(X=x_pca)


sns.pairplot(x_pca, hue='labels')
plt.show()
