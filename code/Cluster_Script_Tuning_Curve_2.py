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
