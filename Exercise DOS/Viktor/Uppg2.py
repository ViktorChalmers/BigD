from matplotlib.colors import Normalize
from tqdm import tqdm,trange
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans, Birch
#from skstab import StadionEstimator
from sklearn.utils import shuffle
from eCDF import*
import pprint


def findFold(fold, point, point2):
    ret = False
    if point in fold and point2 in fold:
        ret = True
    return ret

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
num_folds = 30
length = int(len(x_pca_shuffle)/num_folds) #length of each fold
folds = []
#x_pca_shuffle = np.array(x_pca_shuffle)
"""
for i in range(num_folds-1):
    folds += [x_pca_shuffle[i*length:(i+1)*length]]
folds += [x_pca_shuffle[num_folds-1*length:len(x_pca_shuffle)]]
"""
x_pca=np.array(x_pca)
kluster_list = [3, 4, 5, 6, 7]
C = []
for k in kluster_list:
    for i in range(num_folds):
        indicies = np.random.choice(x_pca.shape[0], size=600, replace=False)
        folds.append(x_pca[indicies,:])

    #model = [KMeans(n_clusters=k).fit(dataset) for dataset in folds]
    model = [Birch(threshold=0.01,branching_factor = 80,n_clusters=k).fit(dataset) for dataset in folds]

    for i in range(len(folds)):
        folds[i] = np.array(folds[i])
    #plt.scatter(folds[0][:, 0], folds[0][:, 1], c=model[0].labels_)
    #plt.show()

    #print(model[0].labels_)

    #print(model[0].predict(folds[1]))
    nrDataPoints = len(x_pca)
    M = [np.zeros([nrDataPoints, nrDataPoints]) for i in range(num_folds)]
    J = [np.zeros([nrDataPoints, nrDataPoints]) for i in range(num_folds)]


    x_pca_shuffle = np.array(x_pca_shuffle)
    i = 3
    j = 1

        #for j in range(len(folds[k])):
         #   if np.array_equal(folds[k][j], point):
          #      for i in range(len(folds[k])):
           #         if np.array_equal(folds[k][i],point2):
            #            return ret

    #print(findFold(folds,x_pca_shuffle[10]))

    #for k in range(num_folds):
    #    modfit = model[k].predict(x_pca_shuffle)
    #    for i in range(801):
    #        for j in range(801):
    #            if modfit[i] == modfit[j]:
    #                M[k][i,j] = 1
    #            if findFold(folds,x_pca_shuffle[i]) == findFold(folds,x_pca_shuffle[j]):
    #                J[k][i,j] = 1



    for k in trange(num_folds):
        modfit = model[k].predict(x_pca)
        for i in range(nrDataPoints):
            for j in range(nrDataPoints):
                if findFold(folds[k], x_pca[i], x_pca[j]):
                    J[k][i, j] = 1
                    if modfit[i] == modfit[j]:
                        M[k][i, j] = 1

    C.append(sum(M)/sum(J))
    print(C)


for ind, Mat in enumerate(C):
    np.save(f'Birch, k={kluster_list[ind]}.npy', Mat)
    visualiseCDF(Mat, label=f'k={kluster_list[ind]}')
plt.show()


