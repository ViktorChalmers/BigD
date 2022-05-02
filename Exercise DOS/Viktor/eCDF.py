import numpy as np
import matplotlib.pyplot as plt


def eCDF(Mat, q):
    Mat_flatten = Mat.flatten()
    F = 0
    for num in Mat_flatten:
        if num <= q:
            F += 1/len(Mat_flatten)
    return F


def visualiseCDF(Mat, label='', num_points=np.linspace(-0.1, 1, 100)):
    Mat_flatten = Mat.flatten()
    F = np.zeros(len(num_points))
    for i, q in enumerate(num_points):
        for num in Mat_flatten:
            if num <= q:
                F[i] += 1 / len(Mat_flatten)
    plt.plot(num_points, F, label=label)
    plt.legend()
    return F

C=np.load('C.npy')
F = eCDF(C, 0.1)
print(F)
visualiseCDF(C, label=f'k=5')
plt.show()
