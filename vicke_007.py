import pandas as pd
import numpy as np
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


prime = optimusPrime(X)
plt.plot(X[:,0],y,"o")
plt.plot(prime[:,0],y,"o")
plt.show()