import pandas as pd 
import numpy as np
from pathlib import Path
import pickle

"""
   utils to be used for importing and wrangling data
"""

animalsAgg = ['29L','3095','3096','3097','30B','30L','30R2','4013','4014','4015','4016','91R2'] # list of all aniamls
animalsObs = ['29L','30R2','86L', '87L2','927L','927R','933R'] # list of observer animals
animalsToy = ['583L2','583B','86L2', '87B', '87L','87R2'] # list of toy group animals
animalsAll = animalsAgg + animalsObs + animalsToy
groupsAll = ['agg' for i in range(len(animalsAgg))] + ['obs' for i in range(len(animalsObs))] + ['toy' for i in range(len(animalsToy))]

# will require Path object
def check_exist(mouseId, group='agg', path=None):
    """
    constructing function that creates path name based on inputs while also checking if the file already exists
    
    params:
    -------
    mouse_id: int or str
        load id of subject mouse or 'all' for all mice together
    
    returns
    -------
    
    """
    # loading path on my hard disk as default
    path = Path("/Volumes/Lenca_SSD/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)

    if group not in ['agg','obs','toy']:
        raise Exception('Group of animal can only be "agg","obs", or "toy"')
    
    # checking that particular mouse_id is a valid option
    if(mouseId  != 'all'):
        if(mouseId not in animalsAll):
            raise Exception('Mouse id can only be "all" or a string in ', animalsAll)
        
    # create a string of the file name to look for
    fname = f"{mouseId}_{group}_neural_data.csv"
    # determine what directory to look for the file in 
    full_path = path / fname

    # return if it exists or not
    return full_path.exists(), full_path

#import Jorge's dataset or wrangle data
def load_and_wrangle(mouseId, group='agg', path=None, overwrite=False):

    """
    Updated function for loading & cleaning individual mouse .csv file
    Data comes from fiber photometry (Ca+2 traces) from either inhibitory or excitatory populations across multipe regions 
    and one DA level

    params:
    -------
    mouseId : str
        id of rat to load
    path : str, default = None (Lenca's local path)
        path where rat_behavior.csv is located, this needs to be a Path object
    overwrite : bool, default=True
        whether to save out and overwrite previously created .csv
    returns
    -------
    df : pandas dataframe
        filtered to specs provided as inputs above
    """
    # loading path on my hard disk as default
    path = Path("/Volumes/Lenca_SSD/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)
    
    exists, full_path = check_exist(mouseId=mouseId, group=group, path=path)
    
    if exists and overwrite==False:
        df = pd.read_csv(full_path)
        return df
    
    else:
        # Load data into dictionary
        with open('../data/fully_labeled_traces_feats3_071924.pickle', 'rb') as handle:
            dict = pickle.load(handle)

        # create dataframe
        dfCol = ['subject','group','other','day','trial']

        if (group == 'agg'):
            rawColumns = dict[f'{mouseId}_d1_balbc_t1'].columns.tolist()
        else: 
            rawColumns = dict[f'{mouseId}_d1_{group}_t1'].columns.tolist()
        dfCol = dfCol + rawColumns
        df = pd.DataFrame(columns = dfCol)

        # hardcode other depending on group
        if (group == 'agg'):
            others = ['balbc', 'mCD1']
        elif (group == 'obs'):
            others = ['OBSmCD1', 'obs']
        elif (group == 'toy'):
            others = ['toy', 'toyCD1']
    
        for key in dict.keys():
            s = key.split('_')
            if (s[0] == mouseId and s[2] in others):
                dfTemp = dict[key]
                dfTemp["subject"] = s[0]
                dfTemp['group'] = group
                dfTemp["other"] = s[2]
                dfTemp["day"] = s[1]
                dfTemp["trial"] = s[3]
                dfTemp = dfTemp[dfCol] # reordering columns
                df = pd.concat([df,dfTemp])
        
        # dropping any row with a missing value within the dataframe
        #df.dropna(inplace = True)
        
        # filtering to add if necessary 

        # resetting index 
        df = df.reset_index(drop=True)
        
        # save out
        df.to_csv(full_path, index = False)

        return df

def get_regions_dataframe(df):
    ''' 
    get regions of neural activity recorded 
    '''

    regions = df.columns.tolist()
    # drop columns that only have missing values
    for col in df.columns:
        if df[col].isna().sum() == len(df.index.tolist()): # dropping columns with only missing values
            regions.remove(col)

    # remove columns        
    regions.remove('day')
    regions.remove('group')
    regions.remove('subject')
    regions.remove('other')
    regions.remove('trial')
    regions.remove('unsupervised labels')

    if 'supervised labels' in regions:
        regions.remove('supervised labels')
    if 'attack labels' in regions:
        regions.remove('attack labels')
    if 'investigation labels' in regions:
        regions.remove('investigation labels')
    
    return regions

def get_design_X_GLM_features(animal, group, features, Nbins=10, path=None):

    ''' 
    Works for multiple features now

    Parameters:
    ----------
    animal: str
        animal ID
    variables: list of str
        behavioral variables to be include in the design matrix
    timelags: list of [int,int]
        both positive


    Returns:
    --------
    X_all: numpy array
        array of behavioral features in time for all days together, except the last day
    X: array of vectors
        X[day] is an array of behavioral features in time given for a particular day
    '''

    # loading path on my hard disk as default
    path = Path("/Volumes/Lenca_SSD/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)
    
    data = load_and_wrangle(mouseId=animal, group=group, path=path, overwrite=False)
    days = np.unique(data['day'])
    trials = np.unique(data['trial'])

    X = np.empty((len(days)), dtype=object) # binned features

    for ind_feature in range(len(features)):
        a = np.empty((len(days)*len(trials)), dtype=object) # all features across sessions 
        c = 0 # counting index
        for ind_day in range(0, len(days)): # day index
            for ind_trial in range(0,len(trials)): # trial index
                temp = data[data['day'] == days[ind_day]]
                temp = temp[temp['trial'] == trials[ind_trial]].reset_index()
                df = pd.read_parquet(f'../data/processed_features_020924_parquets/{animal}_{days[ind_day]}_{temp.loc[0,"other"]}_{trials[ind_trial]}_zscored_features.parquet')
                a[c] = np.array(df[features[ind_feature]])
                c = c + 1

        # creating the bins
        all = np.concatenate(a)
        _, bin_edges = np.histogram(all, bins=Nbins)
            
        # creating arrays with binned features
        c = 0
        for ind_day in range(0, len(days)): # day index
            X_temp = np.empty((len(trials)), dtype=object)
            for ind_trial in range(0,len(trials)): # trial index
                X_temp[ind_trial] = np.zeros((a[c].shape[0], Nbins))
                for ind_bin in range(Nbins):
                    ind_lower = np.argwhere(a[c] >=  bin_edges[ind_bin]).flatten()
                    ind_upper = np.argwhere(a[c] <  bin_edges[ind_bin+1]).flatten()
                    ind_binned = list(set(ind_lower).intersection(set(ind_upper)))
                    X_temp[ind_trial][ind_binned, ind_bin] = 1
                c = c + 1    

            if (ind_feature > 0): 
                X[ind_day] = np.concatenate([X[ind_day], np.concatenate((X_temp), axis=0)], axis=1) # concatenate across trials within a day
            else:
                X[ind_day] = np.concatenate((X_temp), axis=0)

    # add bias term as first coulumns
    for ind_day in range(0, len(days)): # day index
        X[ind_day] = np.concatenate([np.ones((X[ind_day].shape[0], 1)), X[ind_day]], axis=1) 

    X_all = np.concatenate(X[:-1], axis=0) # concatenate all days together, EXCEPT last day
    bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2

    return X_all, X, bin_centers

def get_design_day9_X_GLM_features(animal, group, features, Nbins=10, path=None):

    ''' 
    Works for multiple features now

    Parameters:
    ----------
    animal: str
        animal ID
    variables: list of str
        behavioral variables to be include in the design matrix
    timelags: list of [int,int]
        both positive


    Returns:
    --------
    X_all: numpy array
        array of behavioral features in time for all days together, except the last day
    X: array of vectors
        X[day] is an array of behavioral features in time given for a particular day
    '''

    # loading path on my hard disk as default
    path = Path("/Volumes/Lenca_SSD/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)
    
    data = load_and_wrangle(mouseId=animal, group=group, path=path, overwrite=False)
    trials = np.unique(data['trial'])
    data = data[data['day'] == 'd9']

    for ind_feature in range(len(features)):
        a = np.empty((len(trials)), dtype=object) # all features across sessions 
        c = 0 # counting index
        
        for ind_trial in range(0,len(trials)): # trial index
            temp = data[data['trial'] == trials[ind_trial]].reset_index()
            df = pd.read_parquet(f'../data/processed_features_020924_parquets/{animal}_d9_{temp.loc[0,"other"]}_{trials[ind_trial]}_zscored_features.parquet')
            a[c] = np.array(df[features[ind_feature]])
            c = c + 1

        # creating the bins
        all = np.concatenate(a)
        _, bin_edges = np.histogram(all, bins=Nbins)
            
        # creating arrays with binned features
        c = 0
        X_temp = np.empty((len(trials)), dtype=object)
        for ind_trial in range(0,len(trials)): # trial index
            X_temp[ind_trial] = np.zeros((a[c].shape[0], Nbins))
            for ind_bin in range(Nbins):
                ind_lower = np.argwhere(a[c] >=  bin_edges[ind_bin]).flatten()
                ind_upper = np.argwhere(a[c] <  bin_edges[ind_bin+1]).flatten()
                ind_binned = list(set(ind_lower).intersection(set(ind_upper)))
                X_temp[ind_trial][ind_binned, ind_bin] = 1
            c = c + 1    

        if (ind_feature > 0): 
            X = np.concatenate([X, np.concatenate((X_temp), axis=0)], axis=1) # concatenate across trials within a day
        else:
            X = np.concatenate((X_temp), axis=0)

    # add bias term as first coulumns
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1) 

    bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2

    return X, bin_centers

def get_output_Y_GLM(animal, group, region, path=None):
    ''' 
    function to prepare vector output Y (calcium populationa activity for one specific region) for GLM

    Parameters:
    ----------
    animal: str
        animal ID
    region: str
        region name

    Returns:
    --------
    Y_all: numpy array
        vector of calcium activity of the given region for all sessions together
    Y: array of vectors
        Y[day] is a vector of calcium activity in the region given for a particular session
    '''
    # loading path on my hard disk as default
    path = Path("/Volumes/Lenca_SSD/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)
    
    df = load_and_wrangle(mouseId=animal, group=group, path=path, overwrite=False)
    regions = get_regions_dataframe(df)
    days = np.unique(df['day'])
    trials = np.unique(df['trial'])

    if region not in regions:
        return np.nan, np.nan
    else:
        Y = np.empty((len(days)), dtype=object)

        for ind_day in range(0, len(days)): # day index
            Y_temp = np.empty((len(trials)), dtype=object)
            for ind_trial in range(0,len(trials)): # trial index
                dftemp = df[df['day'] == days[ind_day]]
                dftemp = dftemp[dftemp['trial'] == trials[ind_trial]]
                Y_temp[ind_trial] = np.array(dftemp[region])
            Y[ind_day] = np.concatenate((Y_temp), axis=0)
        Y_all = np.concatenate(Y[:-1], axis=0) 

        return Y_all, Y

def get_observed_all_Y(animal, group, path=None):
    ''' 
    function to get multi-region observed activity Y for each session

    Parameters:
    ----------
    animal: str
        animal ID

    Returns:
    --------
    Y_all: numpy array
        vector of calcium activity of the given region for all sessions together
    Y: array of vectors
        Y[day] is a vector of calcium activity in the region given for a particular session
    '''
    # loading path on my hard disk as default
    path = Path("/Volumes/Lenca_SSD/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)
    
    df = load_and_wrangle(mouseId=animal, group=group, path=path, overwrite=False)
    regions = get_regions_dataframe(df)
    days = np.unique(df['day'])
    trials = np.unique(df['trial'])


    Y = np.empty((len(days)*len(trials)), dtype=object)

    for ind_day in range(0, len(days)): # day index
        for ind_trial in range(0,len(trials)): # trial index

            dftemp = df[df['day'] == days[ind_day]]
            dftemp = dftemp[dftemp['trial'] == trials[ind_trial]]
            dftemp = dftemp.drop(columns=['subject','group','other','day','trial','unsupervised labels', 'attack labels'])
            Y[ind_day* len(trials) + ind_trial] = np.array(dftemp)

    return Y

def index_filter_design_matrices_for_specific_behaviors(animal, group, behavior_label, path=None):
    ''' 
    Currently working for supervised labels only (0 no attack - 1 attack)

    Parameters:
    -----------
        behavior_label: int
            'attack label' in dataframe
    '''
    data = load_and_wrangle(mouseId=animal, group=group, path=path, overwrite=False)
    days = np.unique(data['day'])

    indices_filter = np.empty((len(days)), dtype=object)
    number_frames = 0
    for ind_day in range(0, len(days)): # day index
        temp = data[data['day'] == days[ind_day]].reset_index() # new indices for each day
        temp = temp[temp['attack labels']==behavior_label]
        indices_filter[ind_day] = temp.index.tolist()
    
    temp = data[data['day'] != days[-1]] # all except last day
    temp = temp[temp['attack labels']==behavior_label]
    indices_all = temp.index.tolist()
            
    return indices_all, indices_filter






    