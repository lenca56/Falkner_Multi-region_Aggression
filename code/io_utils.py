import pandas as pd 
import numpy as np
from pathlib import Path
import pickle

"""
   utils to be used for importing and wrangling data
"""

subject_ids = [3095,3096,3097,4013,4014,4015,4016]
rawColumns = ['PrL (E)', 'PrL (I)', 'vLS (E)', 'vLS (I)', 'POA (E)', 'POA (I)',
       'BNST (E)', 'BNST (I)', 'AH (E)', 'AH (I)', 'MeA (E)', 'MeA (I)',
       'VMH (E)', 'VMH (I)', 'NAc (DA)', 'unsupervised labels',
       'supervised labels']

# will require Path object
def check_exist(mouseId, path=None):
    """
    constructing function that creates path name based on inputs while also checking if the file already exists
    params:
    -------
    mouse_id: int or str
        load id of subject mouse 
    
    returns
    -------
    
    """
    # loading path on my laptop as default
    path = Path("/Users/lencacuturela/Desktop/Research/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)

    # checking that particular mouse_id is a valid option
    if (int(mouseId) not in subject_ids):
        raise Exception('Mouse id can only be in ', subject_ids)
    else:
        # create a string of the file name to look for
        fname = f"{mouseId}.csv"
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
        with open('../data/full_traces.pickle', 'rb') as handle:
            dict = pickle.load(handle)

        # create dataframe
        dfCol = ['subject','other','day','trial']
        rawColumns = dict[f'{mouseId}_d1_balbc_t1'].columns.tolist()
        dfCol = dfCol + rawColumns
        df = pd.DataFrame(columns = dfCol)
        
        # load data for mouse
        for key in dict.keys():
            if(int(key[0:4])==int(mouseId)):
                dfTemp = dict[key]
                dfTemp["subject"] = int(mouseId)
                dfTemp["other"] = key[8:12]
                dfTemp["day"] = int(key[6])
                dfTemp["trial"] = int(key[-1])
                dfTemp = dfTemp[dfCol] # reordering columns
                df = pd.concat([df,dfTemp])
        
        # dropping any row with a missing value within the dataframe
        #df.dropna(inplace = True)
        
        # filtering to add if necessary
        
        # save out
        df.to_csv(full_path, index = False)
        
        return df

    