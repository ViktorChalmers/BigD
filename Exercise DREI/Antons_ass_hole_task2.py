import tarfile
import pandas as pd
from sklearn.feature_selection import SelectKBest


try:
    data = pd.read_pickle("data.pkl")
except:
    file = tarfile.open('TCGA-PANCAN-HiSeq-801x20531.tar.gz')
    file.extractall()
    data = pd.read_csv("TCGA-PANCAN-HiSeq-801x20531/data.csv", index_col=0)
    data.to_pickle("data.pkl")

label = list(data.columns.values)
print(label)
selectK = SelectKBest(score_func='f_classif', k=200)
selectK.fit_transform(data, label)
print(data)