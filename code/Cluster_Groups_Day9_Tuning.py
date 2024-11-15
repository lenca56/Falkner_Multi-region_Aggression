# importing modules and packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io_utils import *
from plotting_utils import *
from analysis_utils import * 
from pathlib import Path
import pickle
import scipy
import sys
import os

animalsAgg = ['29L','3095'] #['29L','3095','3096','3097','30B','30L','30R2','4013','4014','4015','4016','91R2'] # list of all aniamls
animalsObs = ['29L','30R2'] #['29L','30R2','86L', '87L2','927L','927R','933R'] # list of observer animals
animalsToy = ['583L2','583B'] # ['583L2','583B','86L2', '87B', '87L','87R2'] # list of toy group animals
animalsAll = animalsAgg + animalsObs + animalsToy
groupsAll = ['agg' for i in range(len(animalsAgg))] + ['obs' for i in range(len(animalsObs))] + ['toy' for i in range(len(animalsToy))]

featuresList = ['proximity'] #["proximity","resident centroid roc 500 ms", "intruder centroid roc 500 ms", 'resident2intruder head-tti','resident2intruder head2head angle', "resident tti2head", "intruder tti2head"]

data_path = '../data'
id = pd.DataFrame(columns=['animal','region']) # in total z=532
z = 0
for ind in range(len(animalsAll)):
    animal = animalsAll[ind]
    group = groupsAll[ind]
    df = load_and_wrangle(mouseId=animal, group=group, path=data_path, overwrite=False)
    regions = get_regions_dataframe(df)
    for region in regions:
        id.loc[z, 'animal'] = animal
        id.loc[z, 'region'] = region
        id.loc[z, 'group'] = group
        z += 1

# read from cluster array in order to get parallelizations
idx = 120 #int(os.environ["SLURM_ARRAY_TASK_ID"]) # check 9, 223,311
animal_without = id.loc[idx,'animal']
region = id.loc[idx,'region']
group_without = id.loc[idx, 'group']
print(animal_without)
print(region)
print(group_without)

# setting hyperparameters
alpha_values = [10,100] #[10**x for x in np.arange(1,5.5,0.5)] 
Nbin = 20
K = 4

Y_all_without = []
X_all_without = np.empty((len(featuresList)), dtype=object)
Y_group_without = []
X_group_without = np.empty((len(featuresList)), dtype=object)


flag_all = np.zeros((len(featuresList)))
flag_group = np.zeros((len(featuresList)))

for ind in range(len(animalsAll)):
    animal = animalsAll[ind]
    group = groupsAll[ind]
    print(animal)

    if animal != animal_without or group != group_without:
        temp_df = load_and_wrangle(mouseId=animal, group=group, path=data_path, overwrite=False)
        temp_df = temp_df[temp_df['day']=='d9'] # only day 9
        temp_regions = get_regions_dataframe(temp_df)
        if region in temp_regions:
            if group == group_without:
                Y_all_without.append(np.array(temp_df[region]))
                Y_group_without.append(np.array(temp_df[region]))
                for ind_feature in range(len(featuresList)):
                    features = [featuresList[ind_feature]]
                    Xtemp, _ = get_design_day9_X_GLM_features(animal, group=group, features=features, Nbins=Nbin, path=data_path)
                    if flag_group[ind_feature] == 0:
                        X_group_without[ind_feature] = np.copy(Xtemp)
                        flag_group[ind_feature] == 1
                    else:
                        X_group_without[ind_feature] = np.concatenate((X_group_without[ind_feature], Xtemp))
                    
                    if flag_all[ind_feature] == 0:
                        X_all_without[ind_feature] = np.copy(Xtemp)
                        flag_all[ind_feature] = 1
                    else:
                        X_all_without[ind_feature] = np.concatenate((X_all_without[ind_feature], Xtemp))
                
            else:
                Y_all_without.append(np.array(temp_df[region]))
                for ind_feature in range(len(featuresList)):
                    features = [featuresList[ind_feature]]
                    Xtemp, _ = get_design_day9_X_GLM_features(animal, group=group, features=features, Nbins=Nbin, path=data_path)

                    if flag_all[ind_feature] == 0:
                        X_all_without[ind_feature] = np.copy(Xtemp)
                        flag_all[ind_feature] = 1
                    else:
                        X_all_without[ind_feature] = np.concatenate((X_all_without[ind_feature], Xtemp))
                

Y_all_without = np.concatenate((Y_all_without))
Y_group_without = np.concatenate((Y_group_without))

W_map_all = np.empty((len(featuresList)), dtype=object)
W_map_group = np.empty((len(featuresList)), dtype=object)
train_mse_all = np.zeros((len(featuresList)))
test_mse_all = np.zeros((len(featuresList)))
r2_animal_test_all = np.zeros((len(featuresList)))
r2_animal_test_group = np.zeros((len(featuresList)))
mse_animal_test_all = np.zeros((len(featuresList)))
mse_animal_test_group = np.zeros((len(featuresList)))

for ind_feature in range(len(featuresList)):

    features = [featuresList[ind_feature]]
    
    # global fit
    W_temp = np.empty((K, len(alpha_values)), dtype=object)
    train_mse_temp = np.zeros((K, len(alpha_values)))
    test_mse_temp = np.zeros((K, len(alpha_values)))
    # Find best alpha from day 9 curve for all animals (since for the toy group that is the only real behavioral data)
    for k in range(K):
        presentTrain, presentTest = split_data(N=Y_all_without.shape[0], Kfolds=K, blocks=400, random_state=42)
        X_train, X_test, Y_train, Y_test = X_all_without[ind_feature][presentTrain[k]], X_all_without[ind_feature][presentTest[k]], Y_all_without[presentTrain[k]], Y_all_without[presentTest[k]]
        alpha_features_before = []
        for alpha_ind in range(len(alpha_values)):
            # regularizer hyperparameter
            alpha = alpha_values[alpha_ind] 
            alpha_features = alpha_features_before + [alpha] # only last feature is being tested with different alpha's
            feature_start = [1 + Nbin * x for x in range(len(features))] # start of features
            # Fit to train data
            W_temp[k, alpha_ind] = solution_linear_Gaussian_smoothing(X_train, Y_train, feature_start=feature_start, alpha_features=alpha_features) # bias term and only one tuning curve
            # MSE
            train_mse_temp[k, alpha_ind] = mse(X_train, Y_train, W_temp[k, alpha_ind])
            test_mse_temp[k, alpha_ind] = mse(X_test, Y_test, W_temp[k, alpha_ind])
    # getting best alpha from fits of all days together
    train_mse_mean = np.mean(train_mse_temp, axis=0)
    test_mse_mean = np.mean(test_mse_temp, axis=0)
    best_alpha_ind = np.unravel_index(np.argmin(test_mse_mean), test_mse_mean.shape)[0]
    best_alpha_all = alpha_values[best_alpha_ind]
    # Fit for all animals without one
    W_map_all[ind_feature] = solution_linear_Gaussian_smoothing(X_all_without[ind_feature], Y_all_without, feature_start=[1], alpha_features=[best_alpha_all]) 
    print(W_map_all[ind_feature])

    # group fit
    W_temp = np.empty((K, len(alpha_values)), dtype=object)
    train_mse_temp = np.zeros((K, len(alpha_values)))
    test_mse_temp = np.zeros((K, len(alpha_values)))

    print(Y_group_without.shape)
    print(X_group_without[ind_feature].shape)
    # Find best alpha from day 9 curve for all animals (since for the toy group that is the only real behavioral data)
    for k in range(K):
        presentTrain, presentTest = split_data(N=Y_group_without.shape[0], Kfolds=K, blocks=400, random_state=42)
        X_train, X_test, Y_train, Y_test = X_group_without[ind_feature][presentTrain[k]], X_group_without[ind_feature][presentTest[k]], Y_group_without[presentTrain[k]], Y_group_without[presentTest[k]]
    
        alpha_features_before = []
        for alpha_ind in range(len(alpha_values)):
            # regularizer hyperparameter
            alpha = alpha_values[alpha_ind] 
            alpha_features = alpha_features_before + [alpha] # only last feature is being tested with different alpha's
            feature_start = [1 + Nbin * x for x in range(len(features))] # start of features
            # Fit to train data
            W_temp[k, alpha_ind] = solution_linear_Gaussian_smoothing(X_train, Y_train, feature_start=feature_start, alpha_features=alpha_features) # bias term and only one tuning curve
            # MSE
            train_mse_temp[k, alpha_ind] = mse(X_train, Y_train, W_temp[k, alpha_ind])
            test_mse_temp[k, alpha_ind] = mse(X_test, Y_test, W_temp[k, alpha_ind])
    # getting best alpha from fits of all days together
    train_mse_mean = np.mean(train_mse_temp, axis=0)
    test_mse_mean = np.mean(test_mse_temp, axis=0)
    best_alpha_ind = np.unravel_index(np.argmin(test_mse_mean), test_mse_mean.shape)[0]
    best_alpha_group = alpha_values[best_alpha_ind]
    # Fit for all animals in the group without one
    W_map_group[ind_feature] = solution_linear_Gaussian_smoothing(X_group_without[ind_feature], Y_group_without, feature_start=[1], alpha_features=[best_alpha_group]) 
    print(W_map_group[ind_feature])
    # testing group and global models on missing animal
    X_animal_test,_ = get_design_day9_X_GLM_features(animal_without, group=group_without, features=features, Nbins=Nbin, path=data_path)
    temp_df = load_and_wrangle(mouseId=animal_without, group=group_without, path=data_path, overwrite=False)
    temp_df = temp_df[temp_df['day']=='d9'] # only day 9
    Y_animal_test = np.array(temp_df[region])
    r2_animal_test_all[ind_feature] = compute_r_squared(X_animal_test, Y_animal_test, W_map_all[ind_feature])
    r2_animal_test_group[ind_feature] = compute_r_squared(X_animal_test, Y_animal_test, W_map_group[ind_feature])
    mse_animal_test_all[ind_feature] = mse(X_animal_test, Y_animal_test, W_map_all[ind_feature])
    mse_animal_test_group[ind_feature] = mse(X_animal_test, Y_animal_test, W_map_group[ind_feature])
               
# saving
np.savez(f'../data/{animal_without}/{animal_without}_{group_without}_test_MAP-estimation_day9_region={region}', W_map_all=W_map_all, W_map_group=W_map_group, best_alpha_all=best_alpha_all, best_alpha_group=best_alpha_group, r2_animal_test_all=r2_animal_test_all, r2_animal_test_group=r2_animal_test_group, mse_animal_test_all=mse_animal_test_all, mse_animal_test_group=mse_animal_test_group)

