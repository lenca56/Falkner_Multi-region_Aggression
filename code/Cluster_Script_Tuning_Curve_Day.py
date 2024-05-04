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
# animalsToy = ['86L2', '87B', '87L','87R2'] # NOT FITTING TOY FOR NOW BCS PARQUETS MISSING

# featuresList = ["proximity","resident centroid roc 500 ms", "intruder centroid roc 500 ms",'resident2intruder head-head', 'resident2intruder head-tti','resident2intruder head2head angle', 'resident2intruder head2tti angle', "intruder2resident head2centroid angle"]
# circularList = [0, 0, 0, 0, 0, 1, 1, 1]
featuresList = ["proximity","resident centroid roc 500 ms", "intruder centroid roc 500 ms", 'resident2intruder head-head', 'resident2intruder head-tti','resident2intruder head2head angle', 'resident2intruder head2tti angle', "intruder2resident head2centroid angle",
   "resident tti2head", "intruder tti2head", "resident tailbase2head angle", "intruder tailbase2head angle"] # potentially add more
circularList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

data_path = '../data'
id = pd.DataFrame(columns=['animal','region']) # in total z=267 for the agg and obs animals
z = 0
for animal in animalsAgg:
    group='agg'
    df = load_and_wrangle(mouseId=animal, group='agg', path=data_path, overwrite=False)
    regions = get_regions_dataframe(df)
    for region in regions:
        id.loc[z, 'animal'] = animal
        id.loc[z, 'region'] = region
        id.loc[z, 'group'] = group
        z += 1
for animal in animalsObs:
    group = 'obs'
    df = load_and_wrangle(mouseId=animal, group=group, path=data_path, overwrite=False)
    regions = get_regions_dataframe(df)
    for region in regions:
        id.loc[z, 'animal'] = animal
        id.loc[z, 'region'] = region
        id.loc[z, 'group'] = group
        z += 1

# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
animal = id.loc[idx,'animal']
region = id.loc[idx,'region']
group = id.loc[idx, 'group']

# setting hyperparameters
alpha_values = [10**x for x in np.arange(1,8.5,0.5)] 
Nbin_values = [20]
K = 5
Ndays = 9

W_map_mean = np.empty((len(featuresList), Ndays), dtype=object)
train_mse_mean = np.zeros((len(featuresList), Ndays))
test_mse_mean = np.zeros((len(featuresList), Ndays))
r2 = np.zeros((len(featuresList), Ndays))

# all day fits
fits = np.load(f'../data/{animal}/{animal}_{group}_KFold={K}_MAP-estimation_region={region}.npz', allow_pickle=True)
best_ind = fits['best_ind']

for ind in range(len(featuresList)):

    # getting best alpha and Nbin from fits of all days together
    Nbin = Nbin_values[best_ind[ind][0]]
    alpha = alpha_values[best_ind[ind][1]]

    # fitting K-fold
    W_map, train_mse, test_mse, r2[ind] = fit_KFold_linear_Gaussian_smoothing_per_day(animal, group, features=[featuresList[ind]], circular_features=[circularList[ind]], region=region, Nbin=Nbin, alpha=alpha, K=K, blocks=400, path=data_path)

    # average of fits
    train_mse_mean[ind] = np.mean(train_mse, axis=0)
    test_mse_mean[ind] = np.mean(test_mse, axis=0)
    W_map_mean[ind] = np.mean(W_map, axis=0)
                         
# saving
np.savez(f'../data/{animal}/{animal}_{group}_KFold={K}_MAP-estimation_per-day_region={region}', W_map=W_map_mean, train_mse=train_mse_mean, test_mse=test_mse_mean, r2_best=r2)

