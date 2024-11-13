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

animalsAgg = ['29L','3095','3096','3097','30B','30L','30R2','4013','4014','4015','4016','91R2'] # list of all aniamls
animalsObs = ['29L','30R2','86L', '87L2','927L','927R','933R'] # list of observer animals
animalsToy = ['583L2','583B','86L2', '87B', '87L','87R2'] # list of toy group animals
animalsAll = animalsAgg + animalsObs + animalsToy
groupsAll = ['agg' for i in range(len(animalsAgg))] + ['obs' for i in range(len(animalsObs))] + ['toy' for i in range(len(animalsToy))]

# featuresList = ["proximity","resident centroid roc 500 ms", "intruder centroid roc 500 ms",'resident2intruder head-head', 'resident2intruder head-tti','resident2intruder head2head angle', 'resident2intruder head2tti angle', "intruder2resident head2centroid angle"]
# circularList = [0, 0, 0, 0, 0, 1, 1, 1]
featuresList = ["proximity","resident centroid roc 500 ms", "intruder centroid roc 500 ms", 'resident2intruder head-tti','resident2intruder head2head angle', "resident tti2head", "intruder tti2head"]

data_path = '../data'
id = pd.DataFrame(columns=['animal','region']) # in total z=399
z = 0
for animal in animalsAgg:
    group='agg'
    df = load_and_wrangle(mouseId=animal, group='agg', path=data_path, overwrite=True)
    regions = get_regions_dataframe(df)
    for region in regions:
        id.loc[z, 'animal'] = animal
        id.loc[z, 'region'] = region
        id.loc[z, 'group'] = group
        z += 1
for animal in animalsObs:
    group = 'obs'
    df = load_and_wrangle(mouseId=animal, group=group, path=data_path, overwrite=True)
    regions = get_regions_dataframe(df)
    for region in regions:
        id.loc[z, 'animal'] = animal
        id.loc[z, 'region'] = region
        id.loc[z, 'group'] = group
        z += 1
for animal in animalsToy:
    group = 'toy'
    df = load_and_wrangle(mouseId=animal, group=group, path=data_path, overwrite=True)
    regions = get_regions_dataframe(df)
    for region in regions:
        id.loc[z, 'animal'] = animal
        id.loc[z, 'region'] = region
        id.loc[z, 'group'] = group
        z += 1
print(z)

# # read from cluster array in order to get parallelizations
# idx =  int(os.environ["SLURM_ARRAY_TASK_ID"]) # check 9, 223,311
# animal = id.loc[idx,'animal']
# region = id.loc[idx,'region']
# group = id.loc[idx, 'group']

# # setting hyperparameters
# alpha_values = [10**x for x in np.arange(1,6.5,0.5)] 
# Nbin = 20
# K = 5
# Ndays = 9
# Nday_last = 8

# W_map = np.empty((len(featuresList), Ndays), dtype=object)
# train_mse = np.zeros((len(featuresList), Ndays))
# test_mse = np.zeros((len(featuresList), Ndays))
# r2 = np.zeros((len(featuresList), Ndays))

# for ind in range(len(featuresList)):

#     features = [featuresList[ind]]
#     Y_all, Y = get_output_Y_GLM(animal, group, region, path=data_path)
#     X_all, X, _ = get_design_X_GLM_features(animal, group=group, features=features, Nbins=Nbin, path=data_path)
#     W_temp = np.empty((K, len(alpha_values)), dtype=object)
#     train_mse_temp = np.zeros((K, len(alpha_values)))
#     test_mse_temp = np.zeros((K, len(alpha_values)))

#     # Find best alpha from day 9 curve for all animals (since for the toy group that is the only real behavioral data)
#     for k in range(K):
#         presentTrain, presentTest = split_data(N=Y[Nday_last].shape[0], Kfolds=K, blocks=400, random_state=42)
#         X_train, X_test, Y_train, Y_test = X[Nday_last][presentTrain[k]], X[Nday_last][presentTest[k]], Y[Nday_last][presentTrain[k]], Y[Nday_last][presentTest[k]]
    
#         alpha_features_before = []
#         for alpha_ind in range(len(alpha_values)):
#             # regularizer hyperparameter
#             alpha = alpha_values[alpha_ind] 
#             alpha_features = alpha_features_before + [alpha] # only last feature is being tested with different alpha's
#             feature_start = [1 + Nbin * x for x in range(len(features))] # start of features
#             # Fit to train data
#             W_temp[k, alpha_ind] = solution_linear_Gaussian_smoothing(X_train, Y_train, feature_start=feature_start, alpha_features=alpha_features) # bias term and only one tuning curve
#             # MSE
#             train_mse_temp[k, alpha_ind] = mse(X_train, Y_train, W_temp[k, alpha_ind])
#             test_mse_temp[k, alpha_ind] = mse(X_test, Y_test, W_temp[k, alpha_ind])

#     # getting best alpha from fits of all days together
#     train_mse_mean = np.mean(train_mse_temp, axis=0)
#     test_mse_mean = np.mean(test_mse_temp, axis=0)
#     best_alpha_ind = np.unravel_index(np.argmin(test_mse_mean), test_mse_mean.shape)[0]
#     best_alpha = alpha_values[best_alpha_ind]

#     # fitting K-fold
#     W_map[ind], train_mse[ind], test_mse[ind], r2[ind] = fit_linear_Gaussian_smoothing_per_day(animal, group, features=[featuresList[ind]], region=region, Nbin=Nbin, alpha=best_alpha, path=data_path)
                         
# # saving
# np.savez(f'../data/{animal}/{animal}_{group}_MAP-estimation_per-day_region={region}', W_map=W_map, train_mse=train_mse, test_mse=test_mse, r2=r2)

