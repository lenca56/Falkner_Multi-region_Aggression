import pandas as pd 
import numpy as np
from pathlib import Path
import pickle

"""
   utils to be used for importing and wrangling data
"""

animalIDs = ['29L','3095','3096','3097','30B','30L','30R2','4013','4014','4015','4016','91R2']

# will require Path object
def check_exist(mouseId, path=None):
    """
    constructing function that creates path name based on inputs while also checking if the file already exists
    
    params:
    -------
    mouse_id: int or str
        load id of subject mouse or 'all' for all mice together
    
    returns
    -------
    
    """
    # loading path on my laptop as default
    path = Path("/Users/lencacuturela/Desktop/Research/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)

    # checking that particular mouse_id is a valid option
    if(mouseId  != 'all'):
        if(mouseId not in animalIDs):
            raise Exception('Mouse id can only be "all" or a string in ', animalIDs)
        
    # create a string of the file name to look for
    fname = f"{mouseId}_neural_data.csv"
    # determine what directory to look for the file in 
    full_path = path / fname

    # return if it exists or not
    return full_path.exists(), full_path

#import Jorge's dataset or wrangle data
def load_and_wrangle(mouseId, path=None, overwrite=False):

    """
    Updated function for loading & cleaning individual mouse .csv file
    Data comes from fiber photometry (Ca+2 traces) from either inhibitory or excitatory populations across multipe regions 
    and one DA leve

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
    
    exists, full_path = check_exist(mouseId=mouseId, path=path)
    
    if exists and overwrite==False:
        df = pd.read_csv(full_path)
        return df
    
    else:
        # Load data into dictionary
        with open('../data/fully_labeled_traces_smoothedLabels_031024.pickle', 'rb') as handle:
            dict = pickle.load(handle)

        # create dataframe
        dfCol = ['subject','other','day','trial']
        if (mouseId == 'all'):
            rawColumns = dict['4015_d1_balbc_t1'].columns.tolist() # mouse 4015 has max number of columns
        else:
            rawColumns = dict[f'{mouseId}_d1_balbc_t1'].columns.tolist()
        dfCol = dfCol + rawColumns
        df = pd.DataFrame(columns = dfCol)
        
        # load data for mouse or all
        if (mouseId=='all'):
            for key in dict.keys():
                dfTemp = dict[key]
                s = key.split('_')
                dfTemp["subject"] = s[0]
                dfTemp["other"] = s[2]
                dfTemp["day"] = s[1]
                dfTemp["trial"] = s[3]
                df = pd.concat([df,dfTemp])
            df = df[dfCol] # reordering columns
        else:
            for key in dict.keys():
                s = key.split('_')
                if (s[0] == mouseId):
                    dfTemp = dict[key]
                    dfTemp["subject"] = s[0]
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
    regions = df.columns.tolist()
    # drop columns that only have missing values
    for col in df.columns:
        if df[col].isna().sum() == len(df.index.tolist()): # dropping columns with only missing values
            regions.remove(col)
            
    regions.remove('day')
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

def get_design_X_GLM_features(animal, features, Nbins=50, path=None):

    ''' 
    ONLY WORKING FOR SINGLE FEATURE NOW

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
        array of behavioral features in time for all days together
    X: array of vectors
        X[day] is an array of behavioral features in time given for a particular day
    '''

    # loading path on my laptop as default
    path = Path("/Users/lencacuturela/Desktop/Research/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)

    df = load_and_wrangle(mouseId=animal, path=path, overwrite=False)
    days = np.unique(df['day'])
    trials = np.unique(df['trial'])

    X = np.empty((len(days)), dtype=object)
    for ind_feature in range(len(features)):
        a = np.empty((len(days)*len(trials)), dtype=object) # all features across sessions to get optimal bin partition
        c = 0 # counting index
        for ind_day in range(0, len(days)): # day index
            for ind_trial in range(0,len(trials)): # trial index
                if (ind_day == 8):
                    other = 'mCD1'
                else:
                    other = 'balbc'
                df = pd.read_parquet(f'../data/{animal}/{animal}_{days[ind_day]}_{other}_{trials[ind_trial]}_zscored_features.parquet')
                a[c] = np.array(df[features[ind_feature]])
                c = c + 1

        all = np.concatenate(a)
        # bin_edges = np.histogram_bin_edges(all, bins='fd') # optimal number of bins with fd method
        _, bin_edges = np.histogram(all, bins=Nbins)
        # print(bin_edges.shape)
        # plt.figure()
        # plt.title(features[ind_feature])
        # plt.hist(all, bins=bin_edges)
        # plt.show()
        
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
            X[ind_day] = np.concatenate((X_temp), axis=0) # concatenate across trials within a day
            X[ind_day] = np.concatenate([np.ones((X[ind_day].shape[0], 1)), X[ind_day]], axis=1) # add bias term
    X_all = np.concatenate(X, axis=0) # concatenate all days together
    bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2

    return X_all, X, bin_centers

def get_output_Y_GLM(animal, region, path=None):
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
    # loading path on my laptop as default
    path = Path("/Users/lencacuturela/Desktop/Research/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)

    df = load_and_wrangle(mouseId=animal, path=path, overwrite=False)
    regions = get_regions_dataframe(df)
    days = np.unique(df['day'])
    trials = np.unique(df['trial'])

    if region not in regions:

        return np.nan, np.nan
    else:
        Y = np.empty((len(days)), dtype=object)

        for ind_day in range(0, len(days)): # day index
            if (ind_day == 8):
                other = 'mCD1'
            else:
                other = 'balbc'
            Y_temp = np.empty((len(trials)), dtype=object)
            for ind_trial in range(0,len(trials)): # trial index
                dftemp = df[df['day'] == days[ind_day]]
                dftemp = dftemp[dftemp['trial'] == trials[ind_trial]]
                dftemp = dftemp[dftemp['other'] == other].reset_index()
                Y_temp[ind_trial] = np.array(dftemp[region])
            Y[ind_day] = np.concatenate((Y_temp), axis=0)
        Y_all = np.concatenate(Y, axis=0) 

        return Y_all, Y





    