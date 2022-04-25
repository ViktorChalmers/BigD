import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier as Tree
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from tabulate import tabulate
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def optimusPrime(X):
    prime = np.zeros(np.shape(X))

    for i in range(len(X[0])):
        prime[:,i] = (X[:,i]-np.mean(X[:,i]))/np.sqrt(np.std(X[:,i]))

    return prime

def split_data(X, y, n=10):
    X_train = [None]*n
    X_test = [None]*n
    y_train = [None]*n
    y_test = [None]*n

    return X_train, X_test, y_train, y_test


def imbalance(f_imbalance):
    #Only works for f_imbalance >= 0.5
    new_y = np.copy(y)
    new_X = np.copy(X)
    if f_zeros < f_imbalance:
        zero_ind = list(np.where(y == 0)[0])
        n_samples = int((len(zeros) - f_imbalance * len(y))/(f_imbalance - 1) + 1)
        for i in range(n_samples):
            ind = random.sample(zero_ind, 2)
            new_sample = 0.95*X[ind[0],:] + 0.05*X[ind[1],:]
            new_X = np.vstack([new_X, new_sample])
            new_y = np.append(new_y,0)
    elif f_imbalance < 0.5:
        None
    else:
        one_ind = list(np.where(y == 1)[0])
        n_samples = int((len(ones) - f_imbalance * len(y))/(f_imbalance - 1) + 1)
        for i in range(n_samples):
            ind = random.sample(one_ind, 2)
            new_sample = 0.95*X[ind[0],:] + 0.05*X[ind[1],:]
            new_X = np.vstack([new_X, new_sample])
            new_y = np.append(new_y,1)
    return new_y, new_X
    
def print_metrics(vec_accuracy, vec_sensitivity, vec_f1, vec_precision, vec_roc_auc , title):
    data = [['Sensitivity',np.mean(vec_sensitivity),np.std(vec_sensitivity)],
        ['Accuracy',np.mean(vec_accuracy),np.std(vec_accuracy)],
        ['F1-Score',np.mean(vec_f1),np.std(vec_f1)],
        ['Precision',np.mean(vec_precision),np.std(vec_precision)],
        ['ROC-AUC',np.mean(vec_roc_auc),np.std(vec_roc_auc)]]

    print('')
    print(title)
    print (tabulate(data, headers=["Metric","Mean", "Standard Deviation"]))
    print('')

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
ones = y[y==1]
zeros = y[y==0]
f_zeros = len(zeros)/len(y)
f_ones = len(ones)/len(y)
#print(f"%Ones: {f_ones},%Zeros: {f_zeros}")






#With or without K-Fold?
#Try without stratisfy

imbalance_frac = [0.5, 0.65, 0.75, 0.85, 0.95]
in_len = len(imbalance_frac)
#clf = QDA()
clf = Tree(random_state=0)
clf_rescaled = Tree(random_state=0)
clf_weighted = Tree(random_state=0,class_weight='balanced')
n_splits = 10
mult = 10

vec_accuracy = np.zeros((mult*n_splits,in_len))
vec_sensitivity = np.zeros((mult*n_splits,in_len))
vec_f1 = np.zeros((mult*n_splits,in_len))
vec_precision = np.zeros((mult*n_splits,in_len))
vec_roc_auc = np.zeros((mult*n_splits,in_len))

vec_accuracy_w = np.zeros((mult*n_splits,in_len))
vec_sensitivity_w = np.zeros((mult*n_splits,in_len))
vec_f1_w = np.zeros((mult*n_splits,in_len))
vec_precision_w = np.zeros((mult*n_splits,in_len))
vec_roc_auc_w = np.zeros((mult*n_splits,in_len))

vec_accuracy_rs = np.zeros((mult*n_splits,in_len))
vec_sensitivity_rs = np.zeros((mult*n_splits,in_len))
vec_f1_rs = np.zeros((mult*n_splits,in_len))
vec_precision_rs = np.zeros((mult*n_splits,in_len))
vec_roc_auc_rs = np.zeros((mult*n_splits,in_len))

for j in range(in_len):
    #Make more zeros/ones to prefered imbalance
    new_y, new_X = imbalance(imbalance_frac[j])

    #f_zeros = len(new_X[new_y==0])/len(new_y)
    #f_ones = len(new_X[new_y==1])/len(new_y)
    #print(f"%Ones: {f_ones},%Zeros: {f_zeros}")

    

    #skf = StratifiedKFold(n_splits = n_splits, shuffle = True)
    skf = KFold(n_splits = n_splits, shuffle = True)
    sm = SMOTE(random_state=0)
    
    for m in range(mult):
        i = m*n_splits
        #for train_index, test_index in skf.split(new_X, new_y):
        for train_index, test_index in skf.split(new_X):
            X_train, X_test = new_X[train_index], new_X[test_index]
            y_train, y_test = new_y[train_index], new_y[test_index]
            X_res, y_res = sm.fit_resample(X_train, y_train)

            clf.fit(X_train, y_train)
            clf_weighted.fit(X_train, y_train)
            clf_rescaled.fit(X_res,y_res)

            y_pred = clf.predict(X_test)
            y_pred_w = clf_weighted.predict(X_test)
            y_pred_rs = clf_rescaled.predict(X_test)

            vec_accuracy[i,j] = accuracy_score(y_test, y_pred) 
            vec_sensitivity[i,j] = recall_score(y_test,y_pred) 
            vec_f1[i,j] = f1_score(y_test,y_pred) 
            vec_precision[i,j] = precision_score(y_test,y_pred) 
            vec_roc_auc[i,j] = roc_auc_score(y_test,y_pred)

            vec_accuracy_w[i,j] = accuracy_score(y_test, y_pred_w) 
            vec_sensitivity_w[i,j] = recall_score(y_test,y_pred_w) 
            vec_f1_w[i,j] = f1_score(y_test,y_pred_w) 
            vec_precision_w[i,j] = precision_score(y_test,y_pred_w) 
            vec_roc_auc_w[i,j] = roc_auc_score(y_test,y_pred_w)

            vec_accuracy_rs[i,j] = accuracy_score(y_test, y_pred_rs) 
            vec_sensitivity_rs[i,j] = recall_score(y_test,y_pred_rs) 
            vec_f1_rs[i,j] = f1_score(y_test,y_pred_rs) 
            vec_precision_rs[i,j] = precision_score(y_test,y_pred_rs) 
            vec_roc_auc_rs[i,j] = roc_auc_score(y_test,y_pred_rs)

            i += 1

    #print_metrics(vec_accuracy, vec_sensitivity, vec_f1, vec_precision, vec_roc_auc , f'Imbalanced: {imbalance_frac[j]*100}% Zeros')
    #print_metrics(vec_accuracy_w, vec_sensitivity_w, vec_f1_w, vec_precision_w, vec_roc_auc_w , f'Weighted')
    #print_metrics(vec_accuracy_rs, vec_sensitivity_rs, vec_f1_rs, vec_precision_rs, vec_roc_auc_rs , f'SMOTE')

plt.errorbar(imbalance_frac,np.mean(vec_accuracy,axis=0),np.std(vec_accuracy,axis=0),label='No Adjustment',capsize=5)
plt.errorbar(imbalance_frac,np.mean(vec_accuracy_w,axis=0),np.std(vec_accuracy_w,axis=0), label = 'Weighted',capsize=5)
plt.errorbar(imbalance_frac,np.mean(vec_accuracy_rs,axis=0),np.std(vec_accuracy_rs,axis=0), label = 'SMOTE',capsize=5)
plt.legend()
plt.xlabel('Ratio of Zeros')
plt.ylabel('Accuracy Score')


plt.figure()
plt.errorbar(imbalance_frac,np.mean(vec_sensitivity,axis=0),np.std(vec_sensitivity,axis=0),label='No Adjustment',capsize=5)
plt.errorbar(imbalance_frac,np.mean(vec_sensitivity_w,axis=0),np.std(vec_sensitivity_w,axis=0), label = 'Weighted',capsize=5)
plt.errorbar(imbalance_frac,np.mean(vec_sensitivity_rs,axis=0),np.std(vec_sensitivity_rs,axis=0), label = 'SMOTE',capsize=5)
plt.legend()
plt.xlabel('Ratio of Zeros')
plt.ylabel('Sensitivity Score')

plt.figure()
plt.errorbar(imbalance_frac,np.mean(vec_f1,axis=0),np.std(vec_f1,axis=0),label='No Adjustment',capsize=5)
plt.errorbar(imbalance_frac,np.mean(vec_f1_w,axis=0),np.std(vec_f1_w,axis=0), label = 'Weighted',capsize=5)
plt.errorbar(imbalance_frac,np.mean(vec_f1_rs,axis=0),np.std(vec_f1_rs,axis=0), label = 'SMOTE',capsize=5)
plt.legend()
plt.xlabel('Ratio of Zeros')
plt.ylabel('F1-Score')

plt.figure()
plt.errorbar(imbalance_frac,np.mean(vec_precision,axis=0),np.std(vec_precision,axis=0),label='No Adjustment',capsize=5)
plt.errorbar(imbalance_frac,np.mean(vec_precision_w,axis=0),np.std(vec_precision_w,axis=0), label = 'Weighted',capsize=5)
plt.errorbar(imbalance_frac,np.mean(vec_precision_rs,axis=0),np.std(vec_precision_rs,axis=0), label = 'SMOTE',capsize=5)
plt.legend()
plt.xlabel('Ratio of Zeros')
plt.ylabel('Precision Score')

plt.figure()
plt.errorbar(imbalance_frac,np.mean(vec_roc_auc,axis=0),np.std(vec_roc_auc,axis=0),label='No Adjustment',capsize=5)
plt.errorbar(imbalance_frac,np.mean(vec_roc_auc_w,axis=0),np.std(vec_roc_auc_w,axis=0), label = 'Weighted',capsize=5)
plt.errorbar(imbalance_frac,np.mean(vec_roc_auc_rs,axis=0),np.std(vec_roc_auc_rs,axis=0), label = 'SMOTE',capsize=5)
plt.legend()
plt.xlabel('Ratio of Zeros')
plt.ylabel('ROC-AUC Score')

plt.show()
#sensitivity important, bc we want few False-Negatives