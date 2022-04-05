import pandas as pd
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import f1_score
import random

def optimusPrime(X):
    prime = np.zeros(np.shape(X))

    for i in range(len(X[0])):
        prime[:,i] = (X[:,i]-np.mean(X[:,i]))/np.sqrt(np.std(X[:,i]))

    return prime

def split_data(X, y, n=10):
    X_train = [None]*n
    X_test = [None]*n
    y_train = [None]*n
    y_test = [None]*n

    return X_train, X_test, y_train, y_test


def imbalance(f_imbalance):
    new_y = np.copy(y)
    new_X = np.copy(X)
    if f_zeros < f_imbalance:
        zero_ind = list(np.where(y == 0)[0])
        n_samples = int((len(zeros) - f_imbalance * len(y))/(f_imbalance - 1) + 1)
        for i in range(n_samples):
            ind = random.sample(zero_ind, 2)
            new_sample = 0.95*X[ind[0],:] + 0.05*X[ind[1],:]
            new_X = np.vstack([new_X, new_sample])
            new_y = np.append(new_y,0)
    else:
        one_ind = list(np.where(y == 1)[0])
        n_samples = int((len(ones) - f_imbalance * len(y))/(f_imbalance - 1) + 1)
        print(n_samples)
        for i in range(n_samples):
            ind = random.sample(one_ind, 2)
            new_sample = 0.95*X[ind[0],:] + 0.05*X[ind[1],:]
            new_X = np.vstack([new_X, new_sample])
            new_y = np.append(new_y,1)
    return new_y, new_X
    

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
f_zeros = len(zeros)/len(y)
f_ones = len(ones)/len(y)
print(f"%Ones: {f_ones},%Zeros: {f_zeros}")


#Make more zeros/ones to prefered imbalance
imbalance_frac = 0.7 #[0.5, 0.65, 0.75, 0.85, 0.95]
new_y, new_X = imbalance(imbalance_frac)

#Make the predictions and stat learning with sklearn for the different cases

    

'''
clf = QDA()

X = optimusPrime(X)
clf2 = tree.DecisionTreeClassifier().fit(X, y)

pipeline = make_pipeline(StandardScaler().fit(X, y), clf2)

predictions = pipeline.predict(X)
f1 = f1_score(y, predictions, average=None)


scores = cross_val_score(pipeline, X, y, cv=10, n_jobs=1)

print('Cross Validation accuracy scores: %s' % scores)

print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

print(f'\n F1 score {f1}')
'''