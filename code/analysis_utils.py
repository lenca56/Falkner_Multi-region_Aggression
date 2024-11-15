# importing modules and packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io_utils import *
from pathlib import Path
import matplotlib as mpl
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score


# def PCAfunction(mouseId, path=None):
#     df = load_and_wrangle(mouseId=mouseId, path=path, overwrite=False)
#     temp = df.drop(columns=['subject','other','day','trial','unsupervised labels','supervised labels'])
    
#     # filtering or averaging for certain days/trials

#     # preparing matrix X
#     x = np.array(temp)
#     print("Variance for each brain region")
#     print(x.mean(axis=0))

#     n_comp=10
#     pca = PCA(n_components=n_comp)
#     y = pca.fit_transform(x)
#     c_explained_variance = np.cumsum(pca.explained_variance_ratio_)
#     plt.title(f'PCA - Mouse {mouseId}')
#     plt.ylabel("total explained variance")
#     plt.xlabel('components')
#     plt.bar(range(1,n_comp+1),c_explained_variance)
#     plt.xticks(range(1,n_comp+1))
#     plt.axhline(0.8,color='black')
#     plt.show()

# Linear Gaussian GLM model 
def solution_linear_Gaussian_smoothing(X, Y, feature_start, alpha_features):
    ''' 
    Smoothing regularizer for tuning curves
    single-variable tuning curves are also regularized (like with ridge)

    Parameters:
    ___________
    X: numpy matrix
        regressors 
    Y: numpy vector
        target (calcium activity of a region)
    feature_start: list of int 
        indices for first coeff of a feature-specific tuning curve (in ascending order), excluding bias term 0 
    alpha: float
        strength of regularization hyperparameter
    '''
    # Checking that the first column is a bias term
    if X[:,0].sum() != X.shape[0] or feature_start[0]!=1:
        raise Exception('First column should have only 1s to reflect the bias term and features start at index 1')

    L_features = np.zeros((len(feature_start)+1, X.shape[1], X.shape[1])) # prior for each feature separately

    feature_start.append(X.shape[1]) # adding last index to signal end for loop

    L_features[0, 0, 0] = 1 # bias term has regularizer hyperparameter 1

    for ind in range(0, len(feature_start)-1):
       
        for t in range(feature_start[ind]+1, feature_start[ind+1]):
            L_features[ind+1, t,t] = 2
            L_features[ind+1, t,t-1] = -1
            L_features[ind+1, t-1,t] = -1
        
        L_features[ind+1, feature_start[ind], feature_start[ind]] = 1 # first coeff in tuning curve
        L_features[ind+1, feature_start[ind+1]-1, feature_start[ind+1]-1] = 1 # last coeff in tuning curve

        if feature_start[ind+1] - feature_start[ind] == 1: # only one point in tuning curve
            L_features[ind+1, feature_start[ind], feature_start[ind]] = 1
        elif feature_start[ind+1] - feature_start[ind] == 2: # only two points in tuning curve
            L_features[ind+1, feature_start[ind], feature_start[ind]] = 1
            L_features[ind+1, feature_start[ind]+1, feature_start[ind]+1] = 1

        L_features[ind+1] = L_features[ind+1] * alpha_features[ind] # multiplying by strength of prior for each feature
    
    L = np.sum(L_features, axis = 0)

    A = X.T @ X + L
    return np.linalg.solve(X.T @ X + L, X.T @ Y) 

def mse(X_true, Y_true, W_map):

    return np.linalg.norm(X_true @ W_map - Y_true) ** 2 / Y_true.shape[0]

def compute_r_squared(X_true, Y_true, W_map):
    '''
    r_squared = 1 - ( || Y - Y_pred || ^ 2 ) /  ( || Y - mean(Y) || ^ 2)
    '''
    Y_pred = X_true @ W_map
    return r2_score(Y_true, Y_pred)

def split_data(N, Kfolds=5, blocks=100, random_state=42):
    ''' 
    splitting data function for cross-validation by giving out indices of test and train
    splits each session into consecutive blocks that randomly go into train and test => each session appears in both train and test

    !Warning: each session must have at least (folds-1) * blocks  trials

    Parameters
    ----------
    x: n x d numpy array
        full design matrix
    y : n x 1 numpy vector 
        full vector of observations with values 0,1,..,C-1
    sessInd: list of int
        indices of each session start, together with last session end + 1
    folds: int
        number of folds to split the data in (test has 1/folds points of whole dataset)
    blocks: int (default = 10)
        blocks of trials to keep together when splitting data (to keep some time dependencies)
    random_state: int (default=1)
        random seed to always get the same split if unchanged

    Returns
    -------
    trainX: list of train_size[i] x d numpy arrays
        trainX[i] has input train data of i-th fold
    trainY: list of train_size[i] numpy arrays
        trainY[i] has output train data of i-th fold
    trainSessInd: list of lists
        trainSessInd[i] have session start indices for the i-th fold of the train data
    testX: // same for test
    '''
   
    # initializing
    presentTrain = np.empty((Kfolds), dtype=object)
    presentTest = np.empty((Kfolds), dtype=object)

    # split session indices into blocks and get session indices for train and test
    yBlock = np.arange(0, N/blocks)
    kf = KFold(n_splits=Kfolds, shuffle=True, random_state=random_state) # shuffle=True and random_state=int for random splitting, otherwise it's consecutive
    for i, (train_blocks, test_blocks) in enumerate(kf.split(yBlock)):
        train_indices = []
        test_indices = []
        for b in yBlock:
            if (b in train_blocks):
                train_indices = train_indices + list(np.arange(b*blocks, min((b+1) * blocks, N)).astype(int))
            elif(b in test_blocks):
                test_indices = test_indices + list(np.arange(b*blocks, min((b+1) * blocks, N)).astype(int))
            else:
                raise Exception("Something wrong with session block splitting")
            
        presentTrain[i] = train_indices # part of training set for fold i
        presentTest[i] = test_indices # part of test set for fold i

    return presentTrain, presentTest

# def find_best_alpha_from_day_9(animal, group, features, region, alpha_values, alpha_features_before=[]):

def fit_KFold_linear_Gaussian_smoothing_all_days(animal, group, features, region, Nbin_values, alpha_values, alpha_features_before=[], behavioral_filter=None, K=5, blocks=100, path=None):
    ''' 
    fitting all days together
    all tuning curves have the same length 
    only last feature in features list has different alpha_values, others have alpha_features_before

    '''
    # loading path on my hard disk as default
    path = Path("/Volumes/Lenca_SSD/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)

    W_map = np.empty((K, len(Nbin_values), len(alpha_values)), dtype=object)
    train_mse = np.zeros((K, len(Nbin_values), len(alpha_values)))
    test_mse = np.zeros((K, len(Nbin_values), len(alpha_values)))

    if (len(features) != 1 + len(alpha_features_before)):
        raise Exception('Alpha for all features except last one should be given a priori')

    Y_all, _ = get_output_Y_GLM(animal, group, region, path=path)
    # filtering for certain behaviors only
    if behavioral_filter is not None:
        indices_all, _ = index_filter_design_matrices_for_specific_behaviors(animal, group, behavior_label=behavioral_filter, path=path)
        Y_all = Y_all[indices_all]
    presentTrain, presentTest = split_data(N=Y_all.shape[0], Kfolds=K, blocks=blocks, random_state=42)
    
    for Nbin_ind in range(len(Nbin_values)):
        Nbin = Nbin_values[Nbin_ind] # number of bins for tuning curves

        X_all, _, _ = get_design_X_GLM_features(animal, group=group, features=features, Nbins=Nbin, path=path)

        # filtering for certain behaviors only
        if behavioral_filter is not None:
            X_all = X_all[indices_all]
        
        # Split data
        for k in range(K):
        
            X_train, X_test, Y_train, Y_test = X_all[presentTrain[k]], X_all[presentTest[k]], Y_all[presentTrain[k]], Y_all[presentTest[k]]

            for alpha_ind in range(len(alpha_values)):
                # regularizer hyperparameter
                alpha = alpha_values[alpha_ind] 
                alpha_features = alpha_features_before + [alpha] # only last feature is being tested with different alpha's
                feature_start = [1 + Nbin * x for x in range(len(features))] # start of features

                # Fit to train data
                W_map[k, Nbin_ind, alpha_ind] = solution_linear_Gaussian_smoothing(X_train, Y_train, feature_start=feature_start, alpha_features=alpha_features) # bias term and only one tuning curve

                # MSE
                train_mse[k, Nbin_ind, alpha_ind] = mse(X_train, Y_train, W_map[k, Nbin_ind, alpha_ind])
                test_mse[k, Nbin_ind, alpha_ind] = mse(X_test, Y_test, W_map[k, Nbin_ind, alpha_ind])

    return W_map, train_mse, test_mse

def fit_linear_Gaussian_smoothing_per_day(animal, group, features, region, Nbin, alpha, path=None):
    ''' 
    fitting each day separately

    for only one feature fits for now

    '''
    # loading path on my hard disk as default
    path = Path("/Volumes/Lenca_SSD/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)

    # total number of days
    Ndays = 9

    W_map = np.empty((Ndays), dtype=object)
    train_mse = np.zeros((Ndays))
    test_mse = np.zeros((Ndays))
    r2 = np.zeros((Ndays))

    _, Y = get_output_Y_GLM(animal, group, region, path=path)
    _, X, _ = get_design_X_GLM_features(animal, group=group, features=features, Nbins=Nbin, path=path)

    if (X.shape[0] != Ndays):
        raise Exception ("Total number of days is not 9 in X")
    
    for ind_day in range(Ndays):
        
        # Fit to train data
        W_map[ind_day] = solution_linear_Gaussian_smoothing(X[ind_day], Y[ind_day], feature_start=[1], alpha_features=[alpha]) # bias term and only one tuning curve

        # MSE
        train_mse[ind_day] = mse(X[ind_day], Y[ind_day], W_map[ind_day])
        test_mse[ind_day] = mse(X[ind_day], Y[ind_day], W_map[ind_day])

        r2[ind_day] = compute_r_squared(X[ind_day], Y[ind_day], W_map[ind_day])

    return W_map, train_mse, test_mse, r2

    


