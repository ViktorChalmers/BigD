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
coeff = np.zeros([5, 200])
for i in trange(100):
    x_train, y_train = resample(new_data, label, n_samples=round(0.95*np.shape(new_data)[0]))
    #clf = LogisticRegression(multi_class='ovr', solver='liblinear', intercept_scaling=10000, C=C, penalty='l1')
    clf = LogisticRegressionCV(multi_class='ovr', solver='liblinear', intercept_scaling=10000, Cs=C_list, penalty='l1',
                               cv=num_folds, scoring='f1')
    clf.fit(x_train, y_train)

    C1 = get_C1(clf, num_folds)
    print(f'C = {clf.C_}, C1 = {C1}')

    for j in range(5):
        coeff[j] += abs(clf.coef_[j])/100
        feature_importance = abs(clf.coef_[j])
        top_ching = [x[0] for x in enumerate(feature_importance) if x[1] > 0]
        print(f'Index for top 5 features for class {j} = {top_ching}')
        for index in top_ching:
            feature_select_list[j][index] += 1


def plot_histo(feature_list, num):
    classes = clf.classes_
    plt.figure()
    plt.bar(np.linspace(1, 200, 200), feature_list, width=0.5)
    if len([x[0] for x in sorted(enumerate(feature_list)) if x[1]==100])>5:
        top_5_largest=[x[0] for x in sorted(enumerate(feature_list)) if x[1]==100]

        while len(top_5_largest) > 5:
            coeff_list = [coeff[num][i] for i in top_5_largest]
            top_5_largest.pop(coeff_list.index(min(coeff_list)))
    else:
        top_5_largest = [x[0] for x in sorted(enumerate(feature_list), key=lambda x: x[1])[-5:]]

    plt.title(classes[num]+'\nTop 5 features: '+str(top_5_largest))
    plt.xlabel('Feature')
    plt.ylabel('Count')
    plt.savefig('class ' + str(num))


def checkList(count_list):
    element = count_list[0]
    check = True
    for item in count_list:
        if element != item:
            check = False
            return check
    return check

for k in range(5):
    plot_histo(feature_select_list[k], k)

plt.show()


