import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def eCDF(Mat, q):
    Mat_flatten = Mat.flatten()
    F = 0
    for num in Mat_flatten:
        if num <= q:
            F += 1/len(Mat_flatten)
    return F


def visualiseCDF(Mat, label='', num_points=np.linspace(-0.1, 1.2, 100)):
    Mat_flatten = Mat.flatten()
    F = np.zeros(len(num_points))
    for i, q in enumerate(num_points):
        for num in Mat_flatten:
            if num <= q:
                F[i] += 1 / len(Mat_flatten)
    plt.plot(num_points, F, label=label)
    plt.legend()
    return F


C_Kmeans = [np.load(f'DATA/k={i}.npy') for i in range(3,8)]
C_Birch = [np.load(f'DATA/Birch, k={i}.npy') for i in range(3,8)]

k = [3, 4, 5, 6, 7]

'''
for i in trange(5):
    visualiseCDF(C_Birch[i], label=f'k={k[i]}')
plt.xlabel('Consensus index')
plt.ylabel('CDF')
plt.title('Consensus matrix CDFs Birch')
plt.show()
'''

PAC = np.zeros(5)
for i in range(5):
    PAC[i] = eCDF(C_Birch[i], q=0.99)-eCDF(C_Birch[i], q=0.01)
plt.plot(k,PAC)
plt.xlabel('k')
plt.ylabel('PAC')
plt.title('PAC score Birch')
plt.show()
