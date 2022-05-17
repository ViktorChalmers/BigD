import tarfile
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def get_C1(clf, num_folds):
    C1=np.zeros(5)
    for i, score in enumerate(clf.scores_):
        cv_mean = np.mean(clf.scores_[score], axis=0)
        cv_std = np.std(clf.scores_[score], axis=0)
        idx_max_mean = np.argmax(cv_mean)
        idx_C1 = np.where(
            (cv_mean <= cv_mean[idx_max_mean] + cv_std[idx_max_mean] / np.sqrt(num_folds)) &
            (cv_mean >= cv_mean[idx_max_mean])
        )[0][0]
        C1[i] = clf.Cs_[idx_C1]
    return C1


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
'''
samples=label[label['Class'] == 'PRAD']
data_class = new_data[]
print(samples)
'''
#x_train, x_test, y_train, y_test = train_test_split(new_data, label, test_size=0.2, random_state=8)

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

'''
x_train, y_train = resample(new_data, label, n_samples=round(0.95*np.shape(new_data)[0]))
clf = LogisticRegressionCV(multi_class='ovr', solver='liblinear', intercept_scaling=10000, Cs=C_list, penalty='l1',
                               cv=5)
clf.fit(x_train, y_train)
C = clf.C_
'''
feature_select_list = np.zeros([5, 200])
num_folds = 5
for i in trange(50):
    #x_train, x_test, y_train, y_test = train_test_split(new_data, label, test_size=0.05, random_state=8)
    x_train, y_train = resample(new_data, label, n_samples=round(0.95*np.shape(new_data)[1]))
    #clf = LogisticRegression(multi_class='ovr', solver='liblinear', intercept_scaling=10000, C=C, penalty='l1')
    clf = LogisticRegressionCV(multi_class='ovr', solver='liblinear', intercept_scaling=10000, Cs=C_list, penalty='l1',
                               cv=num_folds, scoring='f1')
    clf.fit(x_train, y_train)

    C1 = get_C1(clf, num_folds)
    print(f'C = {clf.C_}, C1 = {C1}')
    for j in range(5):
        feature_importance = abs(clf.coef_[j])
        top_5_ching = [x[0] for x in enumerate(feature_importance) if x[1] > 0]
        print(f'Index for top 5 features for class {j} = {top_5_ching}')
        for index in top_5_ching:
            feature_select_list[j][index] += 1


def plot_viktors_dick(feature_list, num):
    plt.figure()
    plt.bar(np.linspace(1, 200, 200), feature_list, width=0.5)
    top_5_largest = [x[0] for x in sorted(enumerate(feature_list), key=lambda x: x[1])[-5:]]
    plt.xticks(top_5_largest, top_5_largest)
    plt.savefig('class ' + str(num))

for k in range(5):
    plot_viktors_dick(feature_select_list[k], k)

#plt.ylabel('Precentage %')
#plt.xlabel('Feature #')
plt.show()


