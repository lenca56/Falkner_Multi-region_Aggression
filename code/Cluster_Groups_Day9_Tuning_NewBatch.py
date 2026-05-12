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

# ---- animal lists ----
animalsAgg = ['29L','3095','3096','3097','30B','30L','30R2','4013','4014','4015','4016','91R2']
animalsObs = ['29L','30R2','86L','87L2','927L','927R','933R','1162B','1185']
animalsToy = ['583L2','583B','86L2','87B','87L','87R2','1162L2','1162R']

animalsAll = animalsAgg + animalsObs + animalsToy
groupsAll  = (['agg']*len(animalsAgg)
             + ['obs']*len(animalsObs)
             + ['toy']*len(animalsToy))

featuresList = [
    "proximity",
    "resident centroid roc 500 ms",
    "intruder centroid roc 500 ms",
    'resident2intruder head-head',
    'resident2intruder head-tti',
    'resident2intruder head2head angle',
    'resident tti2head',
]

data_path = '../data'

# ---- build the (animal, region, group) index table ----
id = pd.DataFrame(columns=['animal','region','group'])
z = 0
for ind in range(len(animalsAll)):
    animal = animalsAll[ind]
    group  = groupsAll[ind]
    df = load_and_wrangle(mouseId=animal, group=group, path=data_path,
                          overwrite=False, newBatch=True)
    regions = get_regions_dataframe(df)
    for region in regions:
        id.loc[z, 'animal'] = animal
        id.loc[z, 'region'] = region
        id.loc[z, 'group']  = group
        z += 1

# ---- pick this job's animal/region from the SLURM array ----
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
animal_without = id.loc[idx, 'animal']
region         = id.loc[idx, 'region']
group_without  = id.loc[idx, 'group']

# ---- hyperparameters ----
alpha_values = [10**x for x in np.arange(-2, 9, 0.5)]
Nbin = 20
K = 4

# ---- shared bin edges ----
def load_bin_edges(path, features):
    """Load shared bin edges from an .npz, keyed by original feature names."""
    def sanitize(f):
        return f.replace(' ', '_').replace('-', '_')
    with np.load(path) as f:
        return {feat: f[sanitize(feat)] for feat in features}

bin_edges = load_bin_edges('../data/shared_bin_edges_d9.npz', featuresList)

# ---- per-feature accumulators ----
# Each entry is a variable-length array (per-feature mask drops different rows)
X_all_without   = [None] * len(featuresList)
Y_all_without   = [None] * len(featuresList)
X_group_without = [None] * len(featuresList)
Y_group_without = [None] * len(featuresList)

# ---- accumulate training data (everyone except the held-out animal) ----
for ind in range(len(animalsAll)):
    animal = animalsAll[ind]
    group  = groupsAll[ind]
    if animal == animal_without and group == group_without:
        continue                                  # skip held-out animal

    temp_df = load_and_wrangle(mouseId=animal, group=group, path=data_path,
                                overwrite=False, newBatch=True)
    temp_df = temp_df[temp_df['day'] == 'd9']
    temp_regions = get_regions_dataframe(temp_df)
    temp_df = zscore_per_session(temp_df, temp_regions, session_col='trial')

    if region not in temp_regions:
        continue
    Y_full = np.array(temp_df[region])             # unmasked Y for this animal

    for ind_feature in range(len(featuresList)):
        features = [featuresList[ind_feature]]
        Xtemp, mask_tmp, _ = get_design_day9_X_GLM_features(
            animal, group=group, features=features,
            bin_edges=bin_edges, path=data_path,
        )
        Ytemp = Y_full[mask_tmp]                   # aligned to Xtemp

        # always append to the "all" pool
        if X_all_without[ind_feature] is None:
            X_all_without[ind_feature] = Xtemp
            Y_all_without[ind_feature] = Ytemp
        else:
            X_all_without[ind_feature] = np.concatenate([X_all_without[ind_feature], Xtemp])
            Y_all_without[ind_feature] = np.concatenate([Y_all_without[ind_feature], Ytemp])

        # also append to the "group" pool if same group
        if group == group_without:
            if X_group_without[ind_feature] is None:
                X_group_without[ind_feature] = Xtemp
                Y_group_without[ind_feature] = Ytemp
            else:
                X_group_without[ind_feature] = np.concatenate([X_group_without[ind_feature], Xtemp])
                Y_group_without[ind_feature] = np.concatenate([Y_group_without[ind_feature], Ytemp])

# ---- held-out animal: per-feature X and masked Y ----
temp_df = load_and_wrangle(mouseId=animal_without, group=group_without,
                            path=data_path, overwrite=False, newBatch=True)
temp_df = temp_df[temp_df['day'] == 'd9']
temp_df = zscore_per_session(temp_df, region, session_col='trial')
Y_full_test = np.array(temp_df[region])

X_animal_test = [None] * len(featuresList)
Y_animal_test = [None] * len(featuresList)
for ind_feature in range(len(featuresList)):
    features = [featuresList[ind_feature]]
    X_tmp, mask_tmp, _ = get_design_day9_X_GLM_features(
        animal_without, group=group_without, features=features,
        bin_edges=bin_edges, path=data_path,
    )
    X_animal_test[ind_feature] = X_tmp
    Y_animal_test[ind_feature] = Y_full_test[mask_tmp]

# ---- result arrays ----
W_map_all            = np.empty((len(featuresList)), dtype=object)
W_map_group          = np.empty((len(featuresList)), dtype=object)
r2_itself_group      = np.zeros(len(featuresList))
r2_animal_test_all   = np.zeros(len(featuresList))
r2_animal_test_group = np.zeros(len(featuresList))
mse_animal_test_all  = np.zeros(len(featuresList))
mse_animal_test_group = np.zeros(len(featuresList))
best_alpha_all_arr   = np.zeros(len(featuresList))
best_alpha_group_arr = np.zeros(len(featuresList))

# ---- per-feature CV alpha selection + final fits ----
for ind_feature in range(len(featuresList)):
    features = [featuresList[ind_feature]]

    # CV splits depend on per-feature N
    presentTrain_all, presentTest_all = split_data(
        N=Y_all_without[ind_feature].shape[0],
        Kfolds=K, blocks=200, random_state=42,
    )
    presentTrain_group, presentTest_group = split_data(
        N=Y_group_without[ind_feature].shape[0],
        Kfolds=K, blocks=200, random_state=42,
    )

    # ===== global ("all") fit: alpha sweep =====
    W_temp        = np.empty((K, len(alpha_values)), dtype=object)
    train_mse_tmp = np.zeros((K, len(alpha_values)))
    test_mse_tmp  = np.zeros((K, len(alpha_values)))
    for k in range(K):
        X_train = X_all_without[ind_feature][presentTrain_all[k]]
        X_test  = X_all_without[ind_feature][presentTest_all[k]]
        Y_train = Y_all_without[ind_feature][presentTrain_all[k]]
        Y_test  = Y_all_without[ind_feature][presentTest_all[k]]
        for alpha_ind, alpha in enumerate(alpha_values):
            feature_start = [1 + Nbin * x for x in range(len(features))]
            W_temp[k, alpha_ind] = solution_linear_Gaussian_smoothing(
                X_train, Y_train,
                feature_start=feature_start, alpha_features=[alpha],
            )
            train_mse_tmp[k, alpha_ind] = mse(X_train, Y_train, W_temp[k, alpha_ind])
            test_mse_tmp[k, alpha_ind]  = mse(X_test,  Y_test,  W_temp[k, alpha_ind])
    best_alpha_all = alpha_values[int(np.argmin(test_mse_tmp.mean(axis=0)))]
    best_alpha_all_arr[ind_feature] = best_alpha_all
    W_map_all[ind_feature] = solution_linear_Gaussian_smoothing(
        X_all_without[ind_feature], Y_all_without[ind_feature],
        feature_start=[1], alpha_features=[best_alpha_all],
    )

    # ===== group fit: alpha sweep =====
    W_temp        = np.empty((K, len(alpha_values)), dtype=object)
    train_mse_tmp = np.zeros((K, len(alpha_values)))
    test_mse_tmp  = np.zeros((K, len(alpha_values)))
    for k in range(K):
        X_train = X_group_without[ind_feature][presentTrain_group[k]]
        X_test  = X_group_without[ind_feature][presentTest_group[k]]
        Y_train = Y_group_without[ind_feature][presentTrain_group[k]]
        Y_test  = Y_group_without[ind_feature][presentTest_group[k]]
        for alpha_ind, alpha in enumerate(alpha_values):
            feature_start = [1 + Nbin * x for x in range(len(features))]
            W_temp[k, alpha_ind] = solution_linear_Gaussian_smoothing(
                X_train, Y_train,
                feature_start=feature_start, alpha_features=[alpha],
            )
            train_mse_tmp[k, alpha_ind] = mse(X_train, Y_train, W_temp[k, alpha_ind])
            test_mse_tmp[k, alpha_ind]  = mse(X_test,  Y_test,  W_temp[k, alpha_ind])
    best_alpha_group = alpha_values[int(np.argmin(test_mse_tmp.mean(axis=0)))]
    best_alpha_group_arr[ind_feature] = best_alpha_group
    W_map_group[ind_feature] = solution_linear_Gaussian_smoothing(
        X_group_without[ind_feature], Y_group_without[ind_feature],
        feature_start=[1], alpha_features=[best_alpha_group],
    )

    # ===== evaluate on held-out animal (per-feature X / Y) =====
    r2_animal_test_all[ind_feature] = compute_r_squared(
        X_animal_test[ind_feature], Y_animal_test[ind_feature], W_map_all[ind_feature])
    r2_animal_test_group[ind_feature] = compute_r_squared(
        X_animal_test[ind_feature], Y_animal_test[ind_feature], W_map_group[ind_feature])
    mse_animal_test_all[ind_feature] = mse(
        X_animal_test[ind_feature], Y_animal_test[ind_feature], W_map_all[ind_feature])
    mse_animal_test_group[ind_feature] = mse(
        X_animal_test[ind_feature], Y_animal_test[ind_feature], W_map_group[ind_feature])

# ---- save ----
np.savez(
    f'../data/{animal_without}/{animal_without}_{group_without}'
    f'_test_MAP-estimation_day9_region={region}',
    W_map_all=W_map_all,
    W_map_group=W_map_group,
    best_alpha_all=best_alpha_all_arr,
    best_alpha_group=best_alpha_group_arr,
    r2_animal_test_all=r2_animal_test_all,
    r2_animal_test_group=r2_animal_test_group,
    mse_animal_test_all=mse_animal_test_all,
    mse_animal_test_group=mse_animal_test_group
)