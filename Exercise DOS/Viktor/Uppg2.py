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
import pprint

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


drop_n_norm = normalize(new_data)

pca = PCA()
pca.fit(drop_n_norm)

# x_pca=pca.transform(drop_n_norm)
# print(f'Singular values ({len(pca.singular_values_)}): \n{pca.singular_values_}')
# print(f"Reducing dimensions from {drop_n_norm.shape} to {x_pca.shape}")

cut_off = 0.025

chungus_len = len(pca.explained_variance_ratio_[pca.explained_variance_ratio_ > cut_off])
print(chungus_len)  # 10 or less (5) gives many nice results

pca = PCA(n_components=chungus_len)
pca.fit(drop_n_norm)
x_pca = pd.DataFrame(pca.transform(drop_n_norm))

#print(len(x_pca))

x_pca_shuffle = shuffle(x_pca)
num_folds = 50
length = int(len(x_pca_shuffle)/num_folds) #length of each fold
folds = []

for i in range(num_folds-1):
    folds += [x_pca_shuffle[i*length:(i+1)*length]]
folds += [x_pca_shuffle[num_folds-1*length:len(x_pca_shuffle)]]
#print(folds)
k = 5
print(len(folds))

model = [KMeans(n_clusters=k).fit(dataset) for dataset in folds]

#print(model[0].labels_)


#print(model[0].predict(folds[1]))

M = [np.zeros([801, 801]) for i in range(num_folds-1)]
J = [np.zeros([801, 801]) for i in range(num_folds-1)]
x_pca_shuffle = np.array(x_pca_shuffle)

for k in range(num_folds-1):
    modfit = model[k].predict(x_pca_shuffle)
    for i in range(801):
        for j in range(801):
            if modfit[i] == modfit[j]:
                M[k][i,j] = 1
            if np.floor(i/(length-1-1e-3)) == np.floor(j/(length-1-1e-3)):
                J[k][i,j] = 1

pp = pprint.PrettyPrinter()
pp.pprint(sum(J))

print(int(85/80))
#print(np.shape(sum(M)))
C = sum(M)/sum(J)
