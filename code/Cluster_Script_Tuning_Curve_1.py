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
id = pd.DataFrame(columns=['animal','region']) # in total z=311 for the agg and obs animals
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

# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
animal = id.loc[idx,'animal']
region = id.loc[idx,'region']
group = id.loc[idx, 'group']

# setting hyperparameters
alpha_values = [10**x for x in np.arange(0,5.5,0.5)] 
Nbin_values = [10, 30]
K = 5

W_map = np.empty((len(featuresList), K, len(Nbin_values), len(alpha_values)), dtype=object)
train_mse = np.zeros((len(featuresList), K, len(Nbin_values), len(alpha_values)))
test_mse = np.zeros((len(featuresList), K, len(Nbin_values), len(alpha_values)))

for ind in range(len(featuresList)):
    # fitting
    W_map[ind, :, :, :], train_mse[ind, :, :, :], test_mse[ind, :, :, :] = fit_KFold_linear_Gaussian_smoothing(animal=animal, group=group, features=[featuresList[ind]], circular_features=[circularList[ind]], region=region, Nbin_values=Nbin_values, alpha_values=alpha_values, K=K, blocks=400, path=data_path)
                                                               
# saving
np.savez(f'../data/{animal}/{animal}_{group}_KFold={K}_MAP-estimation_region={region}', W_map=W_map, train_mse=train_mse, test_mse=test_mse)

