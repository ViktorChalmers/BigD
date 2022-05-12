import tarfile
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import numpy as np


try:
    data = pd.read_pickle("data.pkl")
except:
    file = tarfile.open('TCGA-PANCAN-HiSeq-801x20531.tar.gz')
    file.extractall()
    data = pd.read_csv("TCGA-PANCAN-HiSeq-801x20531/data.csv", index_col=0)
    data.to_pickle("data.pkl")

label = pd.read_csv("TCGA-PANCAN-HiSeq-801x20531/labels.csv", index_col=0)

selectK = SelectKBest(score_func=f_classif, k=200)
new_data = selectK.fit_transform(data, label)
print(new_data)

C_list = np.logspace(-4, 4, 30)
clf = LogisticRegression(multi_class='ovr', solver='liblinear', intercept_scaling=10000, C=C_list[10])
clf.fit(new_data, label)

feature_importance = abs(clf.coef_[0])
top_5_ching = [x[0] for x in sorted(enumerate(feature_importance), key=lambda x: x[1])[-5:]]
print(top_5_ching)
