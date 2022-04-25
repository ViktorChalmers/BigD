import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

#load data in pandafile
try: #load data if pkl exists
   data = pd.read_pickle("data.pkl")
except:
    try: #loading data from csv if extracted and save in pkl
        data = pd.read_csv("TCGA-PANCAN-HiSeq-801x20531/data.csv", index_col=0)
        data.to_pickle("data.pkl")
    except: # extracting reading and saving data in pkl
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

for i in range(0,len(variance)):
    if variance[i] >=  meanVariance: #setting the threshold as mean variance
        variable.append(columns[i])

print(f"Dropping {len(variance)-len(variable)} features < mean variance")

# creating a new dataframe using the above variables
new_data = data[variable]

# first five rows of the new data
print(new_data.head())
print(f"New data size: {new_data.shape}, old data size: {data.shape}")
#variance of variables in new data
print(new_data.head().var())
plt.plot(variance,  "*")
plt.axhline(meanVariance,   color="red")
plt.figure()
plt.axvline(meanVariance,   color="red")
plt.hist(variance,bins = 50)
plt.show()