import numpy as np
from numpy.random import Generator, PCG64
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import confusion_matrix, mean_squared_error
from collections import namedtuple
import matplotlib.pyplot as plt 

#Label ticks
a = np.array([[1,2,1],[0,-1,3],[2,4,0]])
plt.imshow(a,origin='lower',cmap='Reds')
plt.colorbar()
plt.gca().set_xticks([0.1,0.5,1])
plt.gca().set_yticks([100,200,500])
plt.show()