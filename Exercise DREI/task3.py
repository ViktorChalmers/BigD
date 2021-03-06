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

def plot_cs(beta_count,p,title):
    plt.figure()
    plt.bar(np.linspace(1, p, p),beta_count,width=0.5)
    plt.xlabel('Feature #')
    plt.ylabel('Count')
    plt.title(title)

def diff_lambda(seed,p,n,sparsity,n_folds,SNR,M):
    beta_min = np.zeros(shape=(p,))
    beta_lse = np.zeros(shape=(p,))
    beta_3 = np.zeros(shape=(p,))
    k = 1.25

    X, y, beta = simulate_data(n = n, p = p, rng = Generator(PCG64(seed)), sparsity = sparsity, SNR = SNR)
    beta = make_bin(beta)

    for boot in trange(M):
        #Bootstrap data
        X_boot, y_boot = resample(X, y, n_samples=n)

        #Lasso fit and get hyper parameter
        model_min = LassoCV(cv = n_folds).fit(X_boot, y_boot)   #Gives the best prediction quality
        #alpha_min = model_min.alpha_
        alpha_lse = get_alpha_lse(lasso = model_min, n_folds = n_folds)   #Gives the best feature selection
        model_lse = Lasso(alpha = alpha_lse).fit(X_boot,y_boot)
        model_3 = Lasso(alpha = alpha_lse*k).fit(X_boot,y_boot)

        #Count
        beta_min += make_bin(model_min.coef_)
        beta_lse += make_bin(model_lse.coef_)
        beta_3 += make_bin(model_3.coef_)

    plot_cs(beta_lse,p,r'Count-Statistic of $\lambda_{lse}$')
    plot_cs(beta_min,p,r'Count-Statistic of $\lambda_{min}$')
    plot_cs(beta_3,p,r'Count-Statistic of '+f'{k}'+r'$\lambda_{lse}$')
    plt.show()

###############     MAIN    ###############
seed = 1234567
rng = Generator(PCG64())
p = 750
samp = [100, 400, 700] 
n = 600
sparsity_ls = [0.75, 0.9, 0.99]
sparsity = 0.9
n_folds = 5
SNR_ls = [0.1, 1.0, 2.0]
SNR = 2.0
M = 50

#diff_lambda(seed,p,n,sparsity,n_folds,SNR,M)


beta_lse = np.zeros(shape=(p,3))
for i in trange(3):
    #n = samp[i]
    X, y, beta = simulate_data(n = n, p = p, rng = Generator(PCG64(seed)), sparsity = sparsity, SNR = SNR_ls[i])
    beta = make_bin(beta)

    for boot in range(M):
        #Bootstrap data
        X_boot, y_boot = resample(X, y, n_samples=n)

        #Lasso fit and get hyper parameter
        model_min = LassoCV(cv = n_folds).fit(X_boot, y_boot)   #Gives the best prediction quality
        alpha_min = model_min.alpha_
        alpha_lse = get_alpha_lse(lasso = model_min, n_folds = n_folds)   #Gives the best feature selection
        model_lse = Lasso(alpha = alpha_lse).fit(X_boot,y_boot)
        
        #Count
        beta_lse[:,i] += make_bin(model_lse.coef_)

for i in range(3):    
    plot_cs(beta_lse[:,i],p,r'Count-Statistic of $\lambda_{lse}$, '+f'SNR = {SNR_ls[i]}')
plt.show()

