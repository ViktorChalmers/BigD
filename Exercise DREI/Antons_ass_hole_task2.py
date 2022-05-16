import tarfile
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


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

x_train, x_test, y_train, y_test = train_test_split(new_data, label, test_size=0.5, random_state=8)

C_list = np.logspace(-4, 4, 30)
#TODO split data into validation and training, implement LogisticRegressionCV
clf = LogisticRegression(multi_class='ovr', solver='liblinear', intercept_scaling=10000, C=C_list[10], penalty='l1')
clf.fit(x_train, y_train)

feature_importance = abs(clf.coef_[0])
top_5_ching = [x[0] for x in sorted(enumerate(feature_importance), key=lambda x: x[1])[-5:]]
print(f'Index for top 5 features = {top_5_ching}')
feature_select_list = np.zeros(200)
for index in top_5_ching:
    feature_select_list[index] += 1
prediction=clf.predict(x_test)
print(y_test)
print(prediction)
score = clf.score(x_test, y_test)
print(f'score = {score}')
plt.bar(np.linspace(1, 200, 200), feature_select_list, width=5)
plt.show()

