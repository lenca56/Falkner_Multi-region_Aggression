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
    # loading path on my laptop as default
    path = Path("/Users/lencacuturela/Desktop/Research/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)
    
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
    regions = list(df.columns)
    # drop columns that only have missing values
    for col in df.columns:
        if df[col].isna().sum() == len(df.index.tolist()): # dropping columns with only missing values
            regions.remove(col)
    regions.remove('day')
    regions.remove('subject')
    regions.remove('other')
    regions.remove('trial')
    regions.remove('supervised labels')
    regions.remove('unsupervised labels')
        
    return regions


    