import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


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

# Balanced class labels?
ones = np.size(y[y==1])
print(f'Ones : {ones/(np.size(y))} \nZeros: {1-ones/(np.size(y))}')
# Yes, they are balanced. (10% split is critical)

#Nummerical or Categorical Features? Varying scales?
print(X[0,:])
#They are nummerical and quite varying, should be normalized

print(np.arange(0,29))
#Corrolation?
corrolation = np.corrcoef(X.T)
plt.imshow(corrolation)
plt.colorbar()
#Relatively high, many corrolate between 1 and 0.5
#Area and radius measures corrolate

#preprocess


plt.show()