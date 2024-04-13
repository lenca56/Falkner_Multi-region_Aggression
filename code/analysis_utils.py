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


def PCAfunction(mouseId, path=None):
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
def solution_linear_Gaussian_smoothing(X, Y, feature_start, circular, alpha):
    ''' 
    single features are also regularized (like with ridge)

    Parameters:
    ___________
    X: numpy matrix
        regressors 
    Y: numpy vector
        target (calcium activity of a region)
    feature_start: list of int 
        indices for first coeff of a feature-specific tuning curve (in ascending order), including bias at 0
    circular: list of 0s and 1s (same length as feature_start!)
        1 if that tuning curve is circular and needs smoothing at the ends
    alpha: float
        strength of regularization hyperparameter
    '''
    feature_start.append(X.shape[1]) # adding last possible index to signal end of for loop

    L = np.zeros((X.shape[1], X.shape[1]))
    for ind in range(len(feature_start)-1):
        L[feature_start[ind], feature_start[ind]] = 1 # first coeff in tuning curve
        for t in range(feature_start[ind]+1, feature_start[ind+1]):
            L[t,t] = 2
            L[t,t-1] = -1
            L[t-1,t] = -1

        # check if feature is circular and needs smoothing at its ends
        if circular[ind] == 1: # connecting first and last point of tuning curve
            L[feature_start[ind], feature_start[ind]] = 2
            L[feature_start[ind+1]-1, feature_start[ind+1]-1] = 2
            L[feature_start[ind], feature_start[ind+1]-1] = -1
            L[feature_start[ind+1]-1, feature_start[ind]] = -1
        else: 
            L[feature_start[ind], feature_start[ind]] = 1
            L[feature_start[ind+1]-1, feature_start[ind+1]-1] = 1

        if feature_start[ind+1] - feature_start[ind] == 1: # only one point in tuning curve
            L[feature_start[ind], feature_start[ind]] = 1
        elif feature_start[ind+1] - feature_start[ind] == 2: # only two points in tuning curve
            L[feature_start[ind], feature_start[ind]] = 1
            L[feature_start[ind]+1, feature_start[ind]+1] = 1

    return np.linalg.solve(X.T @ X + alpha * L, X.T @ Y) 

def mse(X_true, Y_true, W_map):

    return np.linalg.norm(X_true @ W_map - Y_true) ** 2 / Y_true.shape[0]

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


def fit_CV_linear_Gaussian_smoothing(animal, features, region, Nbin_values, alpha_values, path=None):
    ''' 
    fitting all days together

    for only one feature for now

    '''
    # loading path on my hard disk as default
    path = Path("/Volumes/Lenca_SSD/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)

    W_map = np.empty((len(Nbin_values), len(alpha_values)), dtype=object)
    train_mse = np.zeros((len(Nbin_values), len(alpha_values)))
    test_mse = np.zeros((len(Nbin_values), len(alpha_values)))

    for Nbin_ind in range(len(Nbin_values)):
        Nbin = Nbin_values[Nbin_ind] # number of bins for tuning curve

        X_all, _, _ = get_design_X_GLM_features(animal, features=features, Nbins=Nbin, path=path)
        Y_all, _ = get_output_Y_GLM(animal, region, path=path)

        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

        for alpha_ind in range(len(alpha_values)):
            alpha = alpha_values[alpha_ind] # regularizer hyperparameter
   
            # Fit to train data
            W_map[Nbin_ind, alpha_ind] = solution_linear_Gaussian_smoothing(X_train, Y_train, feature_start=[0, 1], alpha=alpha) # bias term and only one tuning curve

            # MSE
            train_mse[Nbin_ind, alpha_ind] = mse(X_train, Y_train, W_map[Nbin_ind, alpha_ind])
            test_mse[Nbin_ind, alpha_ind] = mse(X_test, Y_test, W_map[Nbin_ind, alpha_ind])

        
        # find best alpha for this number of bins
        # best_alpha = np.argmin(test_mse[Nbin_ind, :])


    return W_map, train_mse, test_mse

    


