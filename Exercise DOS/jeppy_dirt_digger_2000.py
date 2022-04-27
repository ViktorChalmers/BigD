from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('C:/Users/Jesper/OneDrive/Dokument/GitHub/BigD/Exercise DOS/data.csv',index_col=0,low_memory=False)
 
print(f"{data.head()} Head of data \n {data.shape[0]} datapoints, {data.shape[1]} features")
print(f"{data.isnull().sum() / len(data) * 100}, -> we dont have any missing values in the data")
print(data.dtypes)

print("normalizing data")
normalized = normalize(data)
data_scaled = pd.DataFrame(normalized)
variance = data_scaled.var()
columns = data.columns
print(variance.head())
meanVariance = variance.mean()

variable = []

for i in range(0,len(variance)):
    if variance[i] >=  meanVariance: #setting the threshold as mean variance
        variable.append(columns[i])

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
print(f'Singular values ({len(pca.singular_values_)}): \n{pca.singular_values_}')

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

#Where 2 cut dimensions?