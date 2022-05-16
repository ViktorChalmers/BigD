import tarfile
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

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

x_train, x_test, y_train, y_test = train_test_split(new_data, label, test_size=0.2, random_state=8)

C_list = np.logspace(-4, 4, 30)
#TODO split data into validation and training, implement LogisticRegressionCV
'''
#clf = LogisticRegression(multi_class='ovr', solver='liblinear', intercept_scaling=10000, C=C, penalty='l1')
clf = LogisticRegressionCV(multi_class='ovr', solver='liblinear', intercept_scaling=10000, Cs=C_list, penalty='l1', cv=5)
clf.fit(x_train, y_train)

prediction = clf.predict(x_test)
score = clf.score(x_test, y_test) # Mean accuracy of self.predict(X) wrt. y
score_f1 = f1_score(y_test, prediction, average='macro')

print(f'score = {score}\nf1 score = {score_f1}')

C = C_list[6]
'''
feature_select_list = np.zeros(200)

for i in trange(50):
    x_train, x_test, y_train, y_test = train_test_split(new_data, label, test_size=0.2, random_state=8, shuffle=True)
    #clf = LogisticRegression(multi_class='ovr', solver='liblinear', intercept_scaling=10000, C=C, penalty='l1')
    clf = LogisticRegressionCV(multi_class='ovr', solver='liblinear', intercept_scaling=10000, Cs=C_list, penalty='l1',
                               cv=5)
    clf.fit(x_train, y_train)

    feature_importance = abs(clf.coef_[0])
    top_5_ching = [x[0] for x in sorted(enumerate(feature_importance), key=lambda x: x[1])[-5:]]
    print(f'Index for top 5 features = {top_5_ching}')
    for index in top_5_ching:
        feature_select_list[index] += 1

plt.bar(np.linspace(1, 200, 200), feature_select_list*2, width=2)
plt.ylabel('Precentage %')
plt.xlabel('Feature #')
plt.show()

