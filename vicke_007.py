import pandas as pd
import numpy as np
import numpy.random as rand

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import matplotlib.pyplot as plt

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

def optimusPrime(X):
    prime = np.zeros(np.shape(X))

    for i in range(len(X[0])):
        prime[:,i] = (X[:,i]-np.mean(X[:,i]))/np.sqrt(np.std(X[:,i]))
    return prime

def dicksnballs(data_size = 1000):
    ball1 = [[rand.normal(loc=0, scale=2.0, size=None), rand.normal(loc=10, scale=3.0, size=None)] for _ in
              range(data_size)]
    ball2 = [[rand.normal(loc=10, scale=2, size=None), rand.normal(loc=10, scale=3.0, size=None)] for _ in
              range(data_size)]
    dick = [[rand.normal(loc=5, scale=2.0, size=None), rand.normal(loc=5, scale=5.0, size=None)] for _ in
              range(data_size)]


    class1 = np.array(ball1 + ball2)
    class2 = np.array(dick)
    plt.plot(class1[:, 0], class1[:, 1], 'o', label="class1")
    plt.plot(class2[:, 0], class2[:, 1], '*', label="class2")
    plt.legend()
    plt.show()
    return [class1, class2]

dicksnballs()
prime = optimusPrime(X)

#clf2 = tree.DecisionTreeClassifier()
#pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))
#pipeline = make_pipeline(StandardScaler(), clf2)
#predictions = pipeline.predict(X)