import pandas as pd 
import numpy as np
from pathlib import Path

"""
   utils to be used for importing and wrangling data
"""

subject_ids = [3095,3096,3097,4013,4014,4015,4016]

# will require Path object
def check_exist(mouse_id, path=None):
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
    if (int(mouse_id) not in subject_ids):
        raise Exception('Mouse id can only be in ',subject_ids)
    else:
        # create a string of the file name to look for
        fname = f"{mouse_id}.csv"
        # determine what directory to look for the file in 
        full_path = path / fname

        # return if it exists or not
        return full_path.exists(), full_path

    