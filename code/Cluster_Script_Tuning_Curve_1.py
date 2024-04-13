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

animals = ['29L', '91R2'] #['29L','3095','3096','3097','30B','30L','30R2','4013','4014','4015','4016','91R2'] # list of all aniamls
featuresShortlist = ["proximity","resident centroid roc 500 ms", "intruder centroid roc 500 ms",'resident2intruder head-head', 'resident2intruder head-tti','resident2intruder head2head angle', 'resident2intruder head2tti angle', "intruder2resident head2centroid angle"]
circularShortlist = [0, 0, 0, 0, 0, 1, 1, 1]

data_path = '../data'
id = pd.DataFrame(columns=['animal','region']) # in total z=44 for the two animals
z = 0
for animal in animals:
    df = load_and_wrangle(mouseId=animal, path=data_path, overwrite=False)
    regions = get_regions_dataframe(df)
    for region in regions:
        id.loc[z, 'animal'] = animal
        id.loc[z, 'region'] = region
        z += 1

# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
animal = id.loc[idx,'animal']
region = id.loc[idx,'region']

# setting hyperparameters
alpha_values = [10**x for x in range(-3,7)] 
Nbin_values = [2**x for x in range(3,7)]
K = 5

W_map = np.empty((len(featuresShortlist), K, len(Nbin_values), len(alpha_values)), dtype=object)
train_mse = np.zeros((len(featuresShortlist), K, len(Nbin_values), len(alpha_values)))
test_mse = np.zeros((len(featuresShortlist), K, len(Nbin_values), len(alpha_values)))

for ind in range(len(featuresShortlist)):
    # fitting
    W_map[ind, :, :], train_mse[ind,:,:], test_mse[ind,:,:] = fit_KFold_linear_Gaussian_smoothing(animal=animal, features=[featuresShortlist[ind]], circular_features=[circularShortlist[ind]], region=region, Nbin_values=Nbin_values, alpha_values=alpha_values, K=K, blocks=400, path=data_path)
                                                               
# saving
np.savez(f'../data/{animal}/{animal}_KFold={K}_MAP-estimation_region={region}', W_map=W_map, train_mse=train_mse, test_mse=test_mse)

