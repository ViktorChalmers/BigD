import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy.random as rand
from sklearn import tree

def cartData(data_size = 1000):
    ball1 = [[rand.normal(loc=0, scale=2.0, size=None), rand.normal(loc=10, scale=5.0, size=None)] for _ in
              range(int(data_size/4))]
    ball2 = [[rand.normal(loc=10, scale=2, size=None), rand.normal(loc=10, scale=5.0, size=None)] for _ in
              range(int(data_size/4))]
    blob1 = [[rand.normal(loc=5, scale=2.0, size=None), rand.normal(loc=5, scale=5.0, size=None)] for _ in
              range(int(data_size/4))]
    blob2 = [[rand.normal(loc=15, scale=2.0, size=None), rand.normal(loc=5, scale=5.0, size=None)] for _ in
              range(int(data_size/4))]
    class1 = np.array(ball1 + ball2)
    class2 = np.array(blob1+blob2)
    return [class1, class2]


def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    step = 0.02
    Xmin, Xmax = X[:, 0].min() - 10 * step, X[:, 0].max() + 10 * step
    Ymin, Ymax = X[:, 1].min() - 10 * step, X[:, 1].max() + 10 * step
    XX, YY = np.meshgrid(np.arange(Xmin, Xmax, step),
                         np.arange(Ymin, Ymax, step))
    pred = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    pred = pred.reshape(XX.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(XX, YY, pred, cmap=cmap, alpha=0.25)
    plt.contour(XX, YY, pred, colors='k', linewidths=0.7)
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k')

def print_metrics(vec_accuracy, vec_sensitivity, vec_f1, vec_precision, vec_roc_auc, title,plot = False):
    data = [['Sensitivity', np.mean(vec_sensitivity), np.std(vec_sensitivity)],
            ['Accuracy', np.mean(vec_accuracy), np.std(vec_accuracy)],
            ['F1-Score', np.mean(vec_f1), np.std(vec_f1)],
            ['Precision', np.mean(vec_precision), np.std(vec_precision)],
            ['ROC-AUC', np.mean(vec_roc_auc), np.std(vec_roc_auc)]]

    print('\n')
    print(title)
    print(tabulate(data, headers=["Metric", "Mean", "Standard Deviation"]))
    print('\n')
    if plot==True:
        columns = ("Metric", "Mean", "Standard deviation")
        plt.table(cellText=data, colLabels=columns, loc='center')
        ax = plt.gca()

        # hide x-axis
        ax.get_xaxis().set_visible(False)

        # hide y-axis
        ax.get_yaxis().set_visible(False)

'''QDA data
class1=[[rand.normal(loc=0, scale=2.0, size=None),rand.normal(loc=5, scale=2.0, size=None)] for _ in range(data_size)]
class2=[[rand.normal(loc=5, scale=2, size=None),rand.normal(loc=0, scale=1.0, size=None)] for _ in range(data_size)]
'''

class1,class2 = cartData()#Cart data
X = np.append(class1,class2,axis = 0)

y = np.zeros(len(X))
for i in range(len(class1)):
    y[i] = 1

n_splits = 10
vec_accuracy = np.zeros((n_splits, 1))
vec_sensitivity = np.zeros((n_splits, 1))
vec_f1 = np.zeros((n_splits, 1))
vec_precision = np.zeros((n_splits, 1))
vec_roc_auc = np.zeros((n_splits, 1))
clfList = [QDA(), tree.DecisionTreeClassifier()]
for clf in clfList:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    fig = plt.figure(figsize=(10, 2))
    plt.title(clf)
    i = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        vec_accuracy[i] = accuracy_score(y_test, y_pred)
        vec_sensitivity[i] = recall_score(y_test, y_pred)
        vec_f1[i] = f1_score(y_test, y_pred)
        vec_precision[i] = precision_score(y_test, y_pred)
        vec_roc_auc[i] = roc_auc_score(y_test, y_pred)
        i += 1

    print_metrics(vec_accuracy, vec_sensitivity, vec_f1, vec_precision, vec_roc_auc, f'Imbalanced ',plot=True)
    plot_decision_boundary(clf, X, y, cmap='Paired_r')
plt.show()