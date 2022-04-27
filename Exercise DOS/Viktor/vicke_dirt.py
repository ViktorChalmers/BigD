import pandas as pd
import pyexpat
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.cluster as cluster
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#load data in pandafile

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

    print(f"{data.head()} Head of data \n {data.shape[0]} datapoints, {data.shape[1]} features")
    print(f"{data.isnull().sum() / len(data) * 100}, -> we dont have any missing values in the data")
    print(data.dtypes)

    print("normalizing data")
    normalize = normalize(data)
    data_scaled = pd.DataFrame(normalize)
    variance = data_scaled.var()
    columns = data.columns
    print(variance.head())
    meanVariance = variance.mean()

    variable = []

    for i in range(0, len(variance)):
        if variance[i] >= meanVariance:  # setting the threshold as mean variance
            variable.append(columns[i])

    print(f"Dropping {len(variance) - len(variable)} features < mean variance")

    # creating a new dataframe using the above variables
    new_data = data[variable]

    # first five rows of the new data
    print(new_data.head())
    print(f"New data size: {new_data.shape}, old data size: {data.shape}")
    # variance of variables in new data
    print(new_data.head().var())
    plt.plot(variance, "*")
    plt.axhline(meanVariance, color="red")
    plt.figure()
    plt.axvline(meanVariance, color="red")
    plt.hist(variance, bins=50)
    # plt.show()

    print(new_data.describe())

    print(new_data.head())
    print(type(new_data))
    new_data.to_pickle("new_data.pkl")
    return new_data

try:
    df = pd.read_pickle("new_data.pkl")
except:
    df = filterData()
print(df.head())
print(df.describe())
#Bajskod allt ovanför men det funkar att ladda data och filtrera ned till 6k features


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



scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform((df))

optimise_k_means(scaled_data,15) #undersöker optimalt antal cluster

#dimention reduction Princilal component analysis
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
print(f"Reducing dimensions from {scaled_data.shape} to {x_pca.shape}")

#kmeans clustering 5 clusters
kmeans = KMeans(n_clusters=5)
kmeans.fit(scaled_data)

plt.scatter(x_pca[:,0],x_pca[:,1],c=kmeans.labels_)
plt.show()