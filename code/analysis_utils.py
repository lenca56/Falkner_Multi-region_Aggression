# importing modules and packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io_utils import *
from pathlib import Path
import matplotlib as mpl
from pathlib import Path
from sklearn.decomposition import PCA

def PCAfunction(mouseId, path=None, type=None):
    df = load_and_wrangle(mouseId=mouseId, path=path, overwrite=False)
    temp = df.drop(columns=['subject','other','day','trial','unsupervised labels','supervised labels'])
    
    # filtering or averaging for certain days/trials

    # preparing matrix X
    x = np.array(temp)
    print("Variance for each brain region")
    print(x.mean(axis=0))

    n_comp=10
    pca = PCA(n_components=n_comp)
    y = pca.fit_transform(x)
    c_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.title("Total Explained Variance")
    plt.bar(range(1,n_comp+1),c_explained_variance)
    plt.xticks(range(1,n_comp+1))
    plt.axhline(0.8,color='black')
    plt.show()

    


