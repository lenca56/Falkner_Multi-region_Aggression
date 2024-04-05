# importing modules and packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io_utils import *
from pathlib import Path
import matplotlib as mpl
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def PCAfunction(mouseId, path=None, type=None):
    df = load_and_wrangle(mouseId=mouseId, path=path, overwrite=False)
    temp = df.drop(columns=['subject','other','day','trial','unsupervised labels','supervised labels'])
    
    # filtering or averaging for certain days/trials

    # preparing matrix X
    x = np.array(temp)
    print("Variance for each brain region")
    print(x.mean(axis=0))

    n_comp=10
    pca = PCA(n_components=n_comp)
    y = pca.fit_transform(x)
    c_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.title(f'PCA - Mouse {mouseId}')
    plt.ylabel("total explained variance")
    plt.xlabel('components')
    plt.bar(range(1,n_comp+1),c_explained_variance)
    plt.xticks(range(1,n_comp+1))
    plt.axhline(0.8,color='black')
    plt.show()

# Linear Gaussian GLM model 
def solution_linear_Gaussian_smoothing(X, Y, feature_start, alpha):
    ''' 
    first feature index is bias term, which is not regularized

    Parameters:
    featureIndex: list of int
        indices for first coeff of a feature-specific tuning curve (in ascending order), plus the X.shape[1] at the end
    '''
    L = np.zeros((X.shape[1], X.shape[1]))
    for ind in range(len(feature_start)-1):
        L[feature_start[ind], feature_start[ind]] = 1 # first coeff in tuning curve
        for t in range(feature_start[ind]+1, feature_start[ind+1]):
            L[t,t] = 2
            L[t,t-1] = -1
            L[t-1,t] = -1
        if feature_start[ind+1]-1 == feature_start[ind]: # tuning curve has length 1
            L[feature_start[ind+1]-1, feature_start[ind+1]-1] = 0 # last coeff in tuning curve
        else:
            L[feature_start[ind+1]-1, feature_start[ind+1]-1] = 1
 
    return np.linalg.solve(X.T @ X + alpha * L, X.T @ Y) 

def mse(X_true, Y_true, W_map):

    return np.linalg.norm(X_true @ W_map - Y_true) ** 2 / Y_true.shape[0]

def fit_CV_linear_Gaussian_smoothing(animal, features, region, Nbin_values, alpha_values):
    ''' 
    fitting all days together

    for only one feature for now

    '''

    W_map = np.empty((len(Nbin_values), len(alpha_values)), dtype=object)
    train_mse = np.zeros((len(Nbin_values), len(alpha_values)))
    test_mse = np.zeros((len(Nbin_values), len(alpha_values)))

    for Nbin_ind in range(len(Nbin_values)):
        Nbin = Nbin_values[Nbin_ind] # number of bins for tuning curve

        X_all, _, _ = get_design_X_GLM_features(animal, features=features, Nbins=Nbin, path=None)
        Y_all, _ = get_output_Y_GLM(animal, region, path=None)

        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

        for alpha_ind in range(len(alpha_values)):
            alpha = alpha_values[alpha_ind] # regularizer hyperparameter
   
            # Fit to train data
            W_map[Nbin_ind, alpha_ind] = solution_linear_Gaussian_smoothing(X_train, Y_train, feature_start=[0, 1, X_all.shape[1]], alpha=alpha) # bias term and only one tuning curve

            # MSE
            train_mse[Nbin_ind, alpha_ind] = mse(X_train, Y_train, W_map[Nbin_ind, alpha_ind])
            test_mse[Nbin_ind, alpha_ind] = mse(X_test, Y_test, W_map[Nbin_ind, alpha_ind])

        
        # find best alpha for this number of bins
        best_alpha = np.argmin(test_mse[Nbin_ind, :])

        # fig, axes = plt.subplots()
        # axes.plot(W_map[Nbin_ind, best_alpha][1:])
        # axes.set_xlabel(features[0] + ' filter')
        # axes.set_title(region + ' : alpha = ' + str(alpha_values[best_alpha]))
        # plt.show()

        # fig, axes = plt.subplots()
        # axes.plot(np.log10(alpha_values), train_mse[Nbin_ind, :], color='blue', label='train')
        # axes.plot(np.log10(alpha_values), test_mse[Nbin_ind, :], color='orange', label='test')
        # axes.set_ylabel('MSE')
        # axes.set_xlabel('alpha value')
        # axes.set_title(region)
        # axes.legend()
        # plt.show()

    return W_map, train_mse, test_mse

    


