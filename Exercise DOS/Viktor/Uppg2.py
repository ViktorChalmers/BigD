from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans
#from skstab import StadionEstimator
from sklearn.utils import shuffle

data = pd.read_pickle("new_data.pkl")

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
# print(variance.head())
meanVariance = variance.mean()

variable = []

for i in range(0, len(variance)):
    if variance[i] >= meanVariance:  # setting the threshold as mean variance
        variable.append(columns[i])

# print(f"Dropping {len(variance)-len(variable)} features < mean variance")

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

# x_pca=pca.transform(drop_n_norm)
# print(f'Singular values ({len(pca.singular_values_)}): \n{pca.singular_values_}')
# print(f"Reducing dimensions from {drop_n_norm.shape} to {x_pca.shape}")

cut_off = 0.025

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
print(chungus_len)  # 10 or less (5) gives many nice results

pca = PCA(n_components=chungus_len)
pca.fit(drop_n_norm)
x_pca = pd.DataFrame(pca.transform(drop_n_norm))

#print(len(x_pca))

x_pca_shuffle = shuffle(x_pca)

length = int(len(x_pca_shuffle)/10) #length of each fold
folds = []
for i in range(9):
    folds += [x_pca_shuffle[i*length:(i+1)*length]]
folds += [x_pca_shuffle[9*length:len(x_pca_shuffle)]]
#print(folds)
k = 5
model = [KMeans(n_clusters=k).fit(dataset) for dataset in folds]
#print(model[0].labels_)


#print(model[0].predict(folds[1]))

M = np.zeros([9, 80])
consensusList = [np.zeros([9, 80]) for i in range(9)]

for fold in range(9):
    for i in range(9):
        for j in range(80):
            if model[fold].labels_[j] == model[fold].predict(folds[i])[j]:
                consensusList[fold][i,j] = 1

print(consensusList)