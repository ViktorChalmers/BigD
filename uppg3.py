import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


def optimusPrime(X):
    prime = np.zeros(np.shape(X))

    for i in range(len(X[0])):
        prime[:, i] = (X[:, i] - np.mean(X[:, i])) / np.sqrt(np.std(X[:, i]))

    return prime



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


n_splits = 10
vec_accuracy = np.zeros((n_splits, 1))
vec_sensitivity = np.zeros((n_splits, 1))
vec_f1 = np.zeros((n_splits, 1))
vec_precision = np.zeros((n_splits, 1))
vec_roc_auc = np.zeros((n_splits, 1))
clfList = [QDA(), RandomForestClassifier(max_depth=2, random_state=0), tree.DecisionTreeClassifier()]
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
plt.show()