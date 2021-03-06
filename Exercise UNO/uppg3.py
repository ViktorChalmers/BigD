import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree


def optimusPrime(X):
    prime = np.zeros(np.shape(X))

    for i in range(len(X[0])):
        prime[:, i] = (X[:, i] - np.mean(X[:, i])) / np.sqrt(np.std(X[:, i]))

    return prime


def split_data(X, y, n=10):
    X_train = [None] * n
    X_test = [None] * n
    y_train = [None] * n
    y_test = [None] * n

    return X_train, X_test, y_train, y_test


def imbalance(f_imbalance):
    # Only works for f_imbalance >= 0.5
    new_y = np.copy(y)
    new_X = np.copy(X)
    if f_zeros < f_imbalance:
        zero_ind = list(np.where(y == 0)[0])
        n_samples = int((len(zeros) - f_imbalance * len(y)) / (f_imbalance - 1) + 1)
        for i in range(n_samples):
            ind = random.sample(zero_ind, 2)
            new_sample = 0.95 * X[ind[0], :] + 0.05 * X[ind[1], :]
            new_X = np.vstack([new_X, new_sample])
            new_y = np.append(new_y, 0)
    elif f_imbalance < 0.5:
        None
    else:
        one_ind = list(np.where(y == 1)[0])
        n_samples = int((len(ones) - f_imbalance * len(y)) / (f_imbalance - 1) + 1)
        for i in range(n_samples):
            ind = random.sample(one_ind, 2)
            new_sample = 0.95 * X[ind[0], :] + 0.05 * X[ind[1], :]
            new_X = np.vstack([new_X, new_sample])
            new_y = np.append(new_y, 1)
    return new_y, new_X


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



# Load UCI breast cancer dataset with column names and remove ID column

uci_bc_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    sep=",",
    header=None,
    names=[
        "id_number", "diagnosis", "radius_mean",
        "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean",
        "concavity_mean", "concave_points_mean",
        "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se",
        "area_se", "smoothness_se", "compactness_se",
        "concavity_se", "concave_points_se",
        "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst",
        "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst",
        "concavity_worst", "concave_points_worst",
        "symmetry_worst", "fractal_dimension_worst"
    ], ).drop("id_number", axis=1)

y = uci_bc_data.diagnosis.map({"B": 0, "M": 1}).to_numpy()
X = uci_bc_data.drop("diagnosis", axis=1).to_numpy()
ones = y[y == 1]
zeros = y[y == 0]
f_zeros = len(zeros) / len(y)
f_ones = len(ones) / len(y)
# print(f"%Ones: {f_ones},%Zeros: {f_zeros}")

# Make more zeros/ones to prefered imbalance
imbalance_frac = 0.7  # [0.5, 0.65, 0.75, 0.85, 0.95]
new_y, new_X = imbalance(imbalance_frac)

f_zeros = len(new_X[new_y == 0]) / len(new_y)
f_ones = len(new_X[new_y == 1]) / len(new_y)
print(f"%Ones: {f_ones},%Zeros: {f_zeros}")

# With or without K-Fold?
# Try without stratisfy

n_splits = 10
vec_accuracy = np.zeros((n_splits, 1))
vec_sensitivity = np.zeros((n_splits, 1))
vec_f1 = np.zeros((n_splits, 1))
vec_precision = np.zeros((n_splits, 1))
vec_roc_auc = np.zeros((n_splits, 1))
clfList = [QDA(), RandomForestClassifier(max_depth=2, random_state=0), tree.DecisionTreeClassifier()]
for clf in clfList:
    #clf = RandomForestClassifier(max_depth=2, random_state=0)
    #clf = tree.DecisionTreeClassifier()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    fig = plt.figure(figsize=(10, 2))
    plt.title(clf)


    i = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = new_X[train_index], new_X[test_index]
        y_train, y_test = new_y[train_index], new_y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        vec_accuracy[i] = accuracy_score(y_test, y_pred)
        vec_sensitivity[i] = recall_score(y_test, y_pred)
        vec_f1[i] = f1_score(y_test, y_pred)
        vec_precision[i] = precision_score(y_test, y_pred)
        vec_roc_auc[i] = roc_auc_score(y_test, y_pred)
        i += 1

    print_metrics(vec_accuracy, vec_sensitivity, vec_f1, vec_precision, vec_roc_auc, f'Imbalanced {imbalance_frac}',plot=True)
plt.show()
# sensitivity important, bc we want few False-Negatives

# a) No change
# b) Weight observations
# c) Up/Down sampling

'''
clf = QDA()

new_X = optimusPrime(new_X)
clf2 = tree.DecisionTreeClassifier().fit(new_X, new_y)

pipeline = make_pipeline(StandardScaler().fit(new_X, new_y), clf2)

predictions = pipeline.predict(new_X)
f1 = f1_score(new_y, predictions, average=None)


scores = cross_val_score(pipeline, new_X, new_y, cv=10, n_jobs=1)

print('Cross Validation accuracy scores: %s' % scores)

print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

print(f'\n F1 score {f1}')
'''