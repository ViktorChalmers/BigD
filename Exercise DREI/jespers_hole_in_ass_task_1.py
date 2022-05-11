import numpy as np
from numpy.random import Generator, PCG64
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import confusion_matrix, mean_squared_error
from collections import namedtuple

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

def get_metrics(true, pred):
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return Metrics(sensitivity, specificity)


###############     MAIN    ###############

#p = 500 or 1000
#n = [200, 500, 750, 1000]
#Must be a good p/n ratio
#sparsity = [0.75, 0.9, 0.95, 0.99]
#SNR = 2 or 5
#beta_scale = 5 or 10
rng = Generator(PCG64())
p = 750
n = [100, 200, 400, 750]
n_test = 500
sparsity = [0.75, 0.9, 0.95, 0.99]
repeat = 5
n_folds = 5
Metrics = namedtuple('Metrics','sensitivity specificity')

#Compare:
#1. MSE on training and test data
#2. Sensitivity and Specificity on dataset ???
#Make heat map plot? Very cool


#for i, samples in enumerate(n):
    #for j, spar in enumerate(sparsity):
        #for k in range(repeat):
X_train, y_train, beta_train = simulate_data(n = n[0], p = p, rng = rng, sparsity = sparsity[0])
X_test, y_test, beta_test = simulate_data(n = n_test, p = p, rng = rng, sparsity = sparsity[0])
model_min = LassoCV(cv = n_folds).fit(X_train, y_train)   #Gives the best prediction quality
alpha_min = model_min.alpha_
alpha_lse = get_alpha_lse(lasso = model_min, n_folds = n_folds)
model_lse = Lasso(alpha = alpha_lse).fit(X_train,y_train)

#print(model_min.mse_path_)
#print(model_lse.mse_path_)

#Get MSE
#Use LassoCV to get error from mse_path??
mse_min_train = mean_squared_error(y_train, model_min.predict(X_train))
mse_min_test = mean_squared_error(y_test, model_min.predict(X_test))
mse_lse_train = mean_squared_error(y_train, model_lse.predict(X_train))
mse_lse_test = mean_squared_error(y_test, model_lse.predict(X_test))

#print([mse_min_train, mse_min_test, mse_lse_train, mse_lse_test])

#Get preformance measures
beta_train = make_bin(beta_train)
beta_min = make_bin(model_min.coef_)
beta_lse = make_bin(model_lse.coef_)

train_min_met = get_metrics(beta_train,beta_min)
print(train_min_met.sensitivity)
print(train_min_met.specificity)