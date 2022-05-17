import numpy as np
from numpy.random import Generator, PCG64
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import confusion_matrix, mean_squared_error
from collections import namedtuple
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt 
from sklearn.utils import resample
from tqdm import trange

def simulate_data(n, p, rng, *,sparsity=0.95, SNR=2.0, beta_scale=5.0):
    """Simulate data for Project 3, Part 1.

    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of features
    rng : numpy.random.Generator
        Random number generator (e.g. from `numpy.random.default_rng`)
    sparsity : float in (0, 1)
        Percentage of zero elements in simulated regression coefficients
    SNR : positive float
        Signal-to-noise ratio (see explanation above)
    beta_scale : float
        Scaling for the coefficient to make sure they are large

    Returns
    -------
    X : `n x p` numpy.array
        Matrix of features
    y : `n` numpy.array
        Vector of responses
    beta : `p` numpy.array
        Vector of regression coefficients
    """
    X = rng.standard_normal(size=(n, p))
    
    q = int(np.ceil((1.0 - sparsity) * p))
    beta = np.zeros((p,), dtype=float)
    beta[:q] = beta_scale * rng.standard_normal(size=(q,))
    
    sigma = np.sqrt(np.sum(np.square(X @ beta)) / (n - 1)) / SNR

    y = X @ beta + sigma * rng.standard_normal(size=(n,))

    # Shuffle columns so that non-zero features appear
    # not simply in the first (1 - sparsity) * p columns
    idx_col = rng.permutation(p)
    
    return X[:, idx_col], y, beta[idx_col]

def get_alpha_lse(lasso, n_folds):
    cv_mean = np.mean(lasso.mse_path_, axis=1)
    cv_std = np.std(lasso.mse_path_, axis=1)
    idx_min_mean = np.argmin(cv_mean)
    idx_alpha = np.where(
        (cv_mean <= cv_mean[idx_min_mean] + cv_std[idx_min_mean] / np.sqrt(n_folds)) &
        (cv_mean >= cv_mean[idx_min_mean])
    )[0][0]
    return lasso.alphas_[idx_alpha]

def make_bin(beta):
    beta[beta != 0] = 1
    return beta.astype('int32')


###############     MAIN    ###############

rng = Generator(PCG64())
p = 750
n = [100, 200, 500, 750] 
n = 1000

sparsity = [0.75, 0.9, 0.95, 0.99]
sparsity = [0.75,0.9]

n_folds = 5
M = 50

X, y, beta = simulate_data(n = n, p = p, rng = rng, sparsity = 0.99)
beta = make_bin(beta)

beta_min = np.zeros(p)
beta_lse = np.zeros(p)

for boot in trange(M):
    #Bootstrap data
    X_boot, y_boot = resample(X, y, n_samples=n)

    #Rest
    model_min = LassoCV(cv = n_folds).fit(X_boot, y_boot)   #Gives the best prediction quality
    alpha_min = model_min.alpha_
    alpha_lse = get_alpha_lse(lasso = model_min, n_folds = n_folds)   #Gives the best feature selection
    model_lse = Lasso(alpha = alpha_lse).fit(X_boot,y_boot)
    
    beta_min += make_bin(model_min.coef_)
    beta_lse += make_bin(model_lse.coef_)

plt.figure()
plt.bar(np.linspace(1, p, p),beta_min, width=0.5)
plt.xlabel('Feature #')
plt.ylabel('Count')

plt.figure()
plt.bar(np.linspace(1, p, p),beta_lse,width=0.5)
plt.xlabel('Feature #')
plt.ylabel('Count')

plt.show()