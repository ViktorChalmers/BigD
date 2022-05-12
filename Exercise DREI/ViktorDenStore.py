import tarfile
import pandas as pd

try:
    data = pd.read_pickle("data.pkl")
    labels = pd.read_pickle("labels.pkl")
except:
    file = tarfile.open('TCGA-PANCAN-HiSeq-801x20531.tar.gz')
    file.extractall()

    data = pd.read_csv("TCGA-PANCAN-HiSeq-801x20531/data.csv", index_col=0)
    data.to_pickle("data.pkl")

    labels = pd.read_csv("TCGA-PANCAN-HiSeq-801x20531/labels.csv", index_col=0)
    labels.to_pickle("labels.pkl")



print(f"data is loaded as {type(data)} with shape: {data.shape}")

data["Class"] = labels
target = "Class"
print(data.head())
print(data.Class.value_counts())

X = data.loc[:, data.columns != target]
Y = data.loc[:, data.columns == target]
print(X.shape)
print(Y.shape)
