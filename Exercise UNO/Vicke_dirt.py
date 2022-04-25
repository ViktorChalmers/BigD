import pandas as pd
from sklearn.model_selection import KFold

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

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import numpy as np
from sklearn.model_selection import train_test_split

'''
Observation of each class is drawn from a normal distribution (same as LDA).
QDA assumes that each class has its own covariance matrix (different from LDA).
'''


#for i in range(batch_size):
clf = QDA()

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import f1_score


#
# Create an instance of Pipeline
X = optimusPrime(X)
clf2 = tree.DecisionTreeClassifier().fit(X, y)

#pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))
pipeline = make_pipeline(StandardScaler().fit(X, y), clf2)

predictions = pipeline.predict(X)
f1 = f1_score(y, predictions, average=None)

# Pass instance of pipeline and training and test data set
# cv=10 represents the StratifiedKFold with 10 folds

scores = cross_val_score(pipeline, X, y, cv=10, n_jobs=1)

print('Cross Validation accuracy scores: %s' % scores)

print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

print(f'\n F1 score {f1}')
