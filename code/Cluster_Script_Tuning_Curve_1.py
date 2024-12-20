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
animalsObs = ['29L','30R2','86L', '87L2'] # list of observer animals
animalsToy = ['86L2', '87B', '87L','87R2'] # list of toy animals 

# featuresList = ["proximity","resident centroid roc 500 ms", "intruder centroid roc 500 ms",'resident2intruder head-head', 'resident2intruder head-tti','resident2intruder head2head angle', 'resident2intruder head2tti angle', "intruder2resident head2centroid angle"]
# circularList = [0, 0, 0, 0, 0, 1, 1, 1]
featuresList = ["proximity","resident centroid roc 500 ms", "intruder centroid roc 500 ms", 'resident2intruder head-head', 'resident2intruder head-tti','resident2intruder head2head angle', 'resident2intruder head2tti angle', "intruder2resident head2centroid angle",
   "resident tti2head", "intruder tti2head", "resident tailbase2head angle", "intruder tailbase2head angle"] # potentially add more
data_path = '../data'
id = pd.DataFrame(columns=['animal','region']) # in total z=399 for the agg and obs animals
z = 0

group='agg'
for animal in animalsAgg:
    df = load_and_wrangle(mouseId=animal, group='agg', path=data_path, overwrite=False)
    regions = get_regions_dataframe(df)
    for region in regions:
        id.loc[z, 'animal'] = animal
        id.loc[z, 'region'] = region
        id.loc[z, 'group'] = group
        z += 1

group = 'obs'
for animal in animalsObs:
    df = load_and_wrangle(mouseId=animal, group=group, path=data_path, overwrite=False)
    regions = get_regions_dataframe(df)
    for region in regions:
        id.loc[z, 'animal'] = animal
        id.loc[z, 'region'] = region
        id.loc[z, 'group'] = group
        z += 1

group = 'toy'
for animal in animalsToy:
    df = load_and_wrangle(mouseId=animal, group=group, path=data_path, overwrite=False)
    regions = get_regions_dataframe(df)
    for region in regions:
        id.loc[z, 'animal'] = animal
        id.loc[z, 'region'] = region
        id.loc[z, 'group'] = group
        z += 1

# read from cluster array in order to get parallelizations
idx = 0 #int(os.environ["SLURM_ARRAY_TASK_ID"])
animal = id.loc[idx,'animal']
region = id.loc[idx,'region']
group = id.loc[idx, 'group']

# setting hyperparameters
alpha_values = [10**x for x in np.arange(1,6.5,0.5)] 
Nbin_values = [20]
K = 5

W_map_best = np.empty((len(featuresList)), dtype=object)
best_ind = np.empty((len(featuresList)), dtype=object)
r2_best = np.zeros((len(featuresList)))
train_mse_mean = np.zeros((len(featuresList), len(Nbin_values), len(alpha_values)))
test_mse_mean = np.zeros((len(featuresList), len(Nbin_values), len(alpha_values)))

for ind in [0,1]: #range(len(featuresList)): # for each feature separately

    # fitting K-fold
    _, train_mse, test_mse = fit_KFold_linear_Gaussian_smoothing_all_days(animal=animal, group=group, features=[featuresList[ind]], region=region, Nbin_values=Nbin_values, alpha_values=alpha_values, K=K, blocks=400, path=data_path)

    # average of fits across folds
    train_mse_mean[ind] = np.mean(train_mse, axis=0)
    test_mse_mean[ind] = np.mean(test_mse, axis=0)

    # finding hyperparameters for best fit by 5-fold CV
    best_ind[ind] = np.unravel_index(np.argmin(test_mse_mean[ind]), test_mse_mean[ind].shape)
    Nbin_best = Nbin_values[best_ind[ind][0]]
    alpha_best = alpha_values[best_ind[ind][1]]

    # fitting whole dataset with best hyperparameters
    X_all, _, _ = get_design_X_GLM_features(animal, group=group, features=[featuresList[ind]], Nbins=Nbin_best, path=data_path)
    Y_all, _ = get_output_Y_GLM(animal, group, region, path=data_path)
    W_map_best[ind] = solution_linear_Gaussian_smoothing(X_all, Y_all, feature_start=[1],  alpha=alpha_best) # bias term and only one tuning curve

    # compute r-square for best fit
    r2_best[ind] = compute_r_squared(X_all, Y_all, W_map_best[ind])
                                                               
# saving
np.savez(f'../data/{animal}/{animal}_{group}_KFold={K}_MAP-estimation_region={region}', W_map_best=W_map_best, train_mse=train_mse_mean, test_mse=test_mse_mean, best_ind=best_ind, r2_best=r2_best)

