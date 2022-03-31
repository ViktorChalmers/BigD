import pandas as pd
from sklearn.model_selection import KFold

def split_data(X, y, n=10):
    X_train = [None]*n
    X_test = [None]*n
    y_train = [None]*n
    y_test = [None]*n
    '''
    for i in range(n):
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X, y, train_size=1/n, test_size=1/n)
    '''
    #X_train =
    return X_train, X_test, y_train, y_test

# Load UCI breast cancer dataset with column names and remove ID column

uci_bc_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    sep=",",
    header=None,
    names=[
        "id_number", "diagnosis", "radius_mean",
        "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean",
        "concavity_mean","concave_points_mean",
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
    ],).drop("id_number", axis=1)

y = uci_bc_data.diagnosis.map({"B": 0, "M": 1}).to_numpy()
X = uci_bc_data.drop("diagnosis", axis=1).to_numpy()
ones = y[y==1]
zeros = y[y==0]
print(f"%Ones: {len(ones)/len(y)*100},%Zeros: {len(zeros)/len(y)*100}")

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
from sklearn.model_selection import train_test_split

'''
Observation of each class is drawn from a normal distribution (same as LDA).
QDA assumes that each class has its own covariance matrix (different from LDA).
'''
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size=0.1)
#batch_size = 4
##X_train, X_test, y_train, y_test = split_data(X, y, batch_size)
#print(len(y_train))

#for i in range(batch_size):
#    clf = QuadraticDiscriminantAnalysis()
#    clf.fit(X_train[i], y_train[i])



#QDA(priors=None, reg_param=0.0)
#print(clf.predict([X_test[1][1]]))
#print(clf.predict([np.ones(30)]))

kf = KFold(n_splits=2)
kf.get_n_splits(X)
print(kf)
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
print(np.shape(X))
print(np.shape(X_train))
