from turtle import color
import numpy as np
from numpy.random import Generator, PCG64
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import confusion_matrix, mean_squared_error
from collections import namedtuple
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt 

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

def plot_error(er_matrix, title, isError):
    mean_em = np.mean(er_matrix,axis=0)
    plt.imshow(mean_em,cmap='cool')
    plt.colorbar()
    spar_range = range(mean_em.shape[0])
    n_range = range(mean_em.shape[1])
    plt.gca().set_yticks(spar_range)
    plt.gca().set_xticks(n_range)
    for x in spar_range:
        for y in n_range:
            if isError:
                plt.gca().text(y-0.3, x-0.1, f'{int(mean_em[x,y])}', style='italic')
                plt.gca().text(y-0.3, x+0.1, f'({int(np.std(er_matrix[:,x,y]))})', style='italic')
            else:
                plt.gca().text(y-0.3, x-0.1, f'{mean_em[x,y]:.3f}', style='italic')
                plt.gca().text(y-0.3, x+0.1, f'({np.std(er_matrix[:,x,y]):.4f})', style='italic')
    plt.gca().set_yticklabels(sparsity)
    plt.gca().set_xticklabels(n)
    plt.ylabel('Sparsity')
    plt.xlabel('Samples')
    plt.title(title)
    plt.show()

def plot_sctr(sens,spec,title):
    mean_sens = np.mean(sens,axis=0).flatten()
    mean_spec = np.mean(spec,axis=0).flatten()
    #print(np.mean(sens,axis=0))
    #print(mean_sens)
    markers = ["." , "," , "o" , "v"]  #Must match sparsity size
    colors = ['r','g' ,'b','c']  #Must match n size
    for i,co in enumerate(colors):
        for j,mi in enumerate(markers):
            indx = i+j*len(colors)
            plt.scatter(mean_sens[indx],mean_spec[indx],marker=mi,color=co)
            wid =  np.std(sens[:,j,i])
            hig = np.std(spec[:,j,i])
            circle = Ellipse((mean_sens[indx], mean_spec[indx]), width = wid , height = hig, color=co,alpha = 0.1)
            plt.gca().add_patch(circle)
    for i,col in enumerate(colors):
        plt.scatter([],[],color=col,label=f'Sample = {n[i]}')
    for i,mi in enumerate(markers):
        plt.scatter([],[],color='k',marker=mi,label=f'Sparsity = {sparsity[i]}')
    plt.legend()
    #plt.gca().legend(handles=[red_patch,green_patch])
    plt.xlabel('Sensitivity')
    plt.ylabel('Specificity')
    plt.title(title)
    plt.show()

    

###############     MAIN    ###############

#p = 500 or 1000
#n = [200, 500, 750, 1000]
#Must be a good p/n ratio
#sparsity = [0.75, 0.9, 0.95, 0.99]
#SNR = 2 or 5
#beta_scale = 5 or 10
rng = Generator(PCG64())
p = 750
n = [100, 200, 500, 750] 
#n = [100,200]
n_test = 500
sparsity = [0.75, 0.9, 0.95, 0.99]
#sparsity = [0.75,0.9]
repeat = 5
n_folds = 5
Metrics = namedtuple('Metrics','sensitivity specificity')

#scatter plt on sens n spec

#Compare:
#1. MSE on training and test data
#2. Sensitivity and Specificity on dataset train
#Make heat map plot? Very cool

mse_min_train = np.zeros((repeat,len(sparsity),len(n)))
mse_min_test = np.zeros((repeat,len(sparsity),len(n)))
mse_lse_train = np.zeros((repeat,len(sparsity),len(n)))
mse_lse_test = np.zeros((repeat,len(sparsity),len(n)))

sens_min = np.zeros((repeat,len(sparsity),len(n)))
spec_min = np.zeros((repeat,len(sparsity),len(n)))
sens_lse = np.zeros((repeat,len(sparsity),len(n)))
spec_lse = np.zeros((repeat,len(sparsity),len(n)))

for i, samples in enumerate(n):
    for j, spar in enumerate(sparsity):
        print(f'Sample = {samples} and Sparsity = {spar}')
        for k in range(repeat):
            X_train, y_train, beta_train = simulate_data(n = samples, p = p, rng = rng, sparsity = spar)
            X_test, y_test, beta_test = simulate_data(n = n_test, p = p, rng = rng, sparsity = spar)
            model_min = LassoCV(cv = n_folds).fit(X_train, y_train)   #Gives the best prediction quality
            alpha_min = model_min.alpha_
            alpha_lse = get_alpha_lse(lasso = model_min, n_folds = n_folds)
            model_lse = Lasso(alpha = alpha_lse).fit(X_train,y_train)

            #Get MSE
            #Use LassoCV to get error from .mse_path_??
            mse_min_train[k,j,i] = mean_squared_error(y_train, model_min.predict(X_train))
            mse_min_test[k,j,i] = mean_squared_error(y_test, model_min.predict(X_test))
            mse_lse_train[k,j,i] = mean_squared_error(y_train, model_lse.predict(X_train))
            mse_lse_test[k,j,i] = mean_squared_error(y_test, model_lse.predict(X_test))

            #Get preformance measures
            #Only for test data??
            beta_train = make_bin(beta_train)
            beta_min = make_bin(model_min.coef_)
            beta_lse = make_bin(model_lse.coef_)

            min_met = get_metrics(beta_train,beta_min)
            lse_met = get_metrics(beta_train,beta_lse)
     
            sens_min[k,j,i] = min_met.sensitivity
            spec_min[k,j,i] = min_met.specificity
            sens_lse[k,j,i] = lse_met.sensitivity
            spec_lse[k,j,i] = lse_met.specificity

plot_error(mse_min_train,r'Train MSE of $\lambda_{min}$',True)
plot_error(mse_min_test,r'Test MSE of $\lambda_{min}$',True)
plot_error(mse_lse_train,r'Train MSE of $\lambda_{lse}$',True)
plot_error(mse_lse_test,r'Test MSE of $\lambda_{lse}$',True)
'''
plot_error(sens_min,r'Sensitivity of $\lambda_{min}$',False)
plot_error(spec_min,r'Specificity of $\lambda_{min}$',False)
plot_error(sens_lse,r'Sensitivity of $\lambda_{lse}$',False)
plot_error(spec_lse,r'Specificity of $\lambda_{lse}$',False)
'''
plot_sctr(sens_min,spec_min,r'Metric scatter of $\lambda_{min}$')
plot_sctr(sens_lse,spec_lse,r'Metric scatter of $\lambda_{lse}$')