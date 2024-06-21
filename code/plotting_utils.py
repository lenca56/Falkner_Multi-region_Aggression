# importing modules and packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io_utils import *
from pathlib import Path
import matplotlib as mpl
from pathlib import Path

cmapB = mpl.cm.Blues(np.linspace(0,1,20))
cmapR = mpl.cm.Reds(np.linspace(0,1,20))

def singleFrameBehavioralDistribution(mouseId, path=None, daysType='all',labeltype='unsupervised'):
    """
    plotting single frame distribution of behavioral classes

    params:
    -------
    mouse_id: int or str
        load id of subject mouse or 'all' for all mice together
    type: str
        'supervised' or 'unsupervised' or 'raw' - related to class labels to look at
    
    returns
    -------
    
    """
    # loading path on my laptop as default
    path = Path("/Volumes/Lenca_SSD/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)

    # checking that particular mouse_id is a valid option
    if(mouseId!='all'):
        if(int(mouseId) not in subject_ids):
            raise Exception('Mouse id can only be "all" or in ', subject_ids)

    # checking that type of labels is a valid option
    if (labeltype not in ['raw','supervised','unsupervised']):
        raise Exception('Label type can only be in', ['raw','supervised','unsupervised'])

    # checking that type of day arranging is a valid option
    if (daysType not in ['all days','per day']):
        raise Exception('Dyas type can only be in', ['all days','per day'])

    # info about which class is aggressive behavior
    if(labeltype == 'unsupervised'):
        aggClass = 2
        index_aggClass = 1 # because unsupervised start from 1
    elif(labeltype == 'supervised'):
        aggClass = 3 
        index_aggClass = 3 # because unsupervised start from 0

    # loading data
    df = load_and_wrangle(mouseId=mouseId, path=path, overwrite=False)

    # getting possible classes list
    classes = df[f'{labeltype} labels'].unique()
    classes.sort()

    if (daysType == 'all days'):
        # plotting distribution of classes across all days 

        # scaling for total number of frames
        totalFrames = len(df.index.tolist())

        plt.title("Mouse " + str(mouseId))
        plt.bar(classes,df[f'{labeltype} labels'].value_counts()[classes]/totalFrames,color='gray')
        plt.bar(aggClass,df[f'{labeltype} labels'].value_counts()[aggClass]/totalFrames,color='darkred',label='agg') # coloring aggression red
        plt.ylabel('single frame count')
        plt.xlabel(f'{labeltype} classes')
        plt.xticks(classes)
        plt.legend()
        plt.show()

    elif (daysType == 'per day'):
        # plotting distribution of classes for each day
        days = df['day'].unique()
        N = 3
        ind = np.arange(0,len(classes)) 
        width = 0.1
        for day in days:
            temp = df[df['day']==day]

            # scaling for number of frames per day
            totalFrames = len(temp.index.tolist())

            plt.bar(ind + width*(day-1), temp[f'{labeltype} labels'].value_counts()[classes]/totalFrames, width, color=cmapB[day*2],label='day '+str(day))
            plt.bar(ind[index_aggClass] + width*(day-1), temp[f'{labeltype} labels'].value_counts()[aggClass]/totalFrames, width, color=cmapR[day*2])
        plt.xlabel(f'{labeltype} classes')
        plt.xticks(ticks = ind + width*4,labels = classes)
        plt.ylabel('single frame count')
        plt.legend()
        plt.title(f'Mouse {mouseId}')
        plt.show()

def behaviorDistributions(mouseId,path=None,type='unsupervised'):
    """
    plotting different behavioral distributions 
    params:
    -------
    mouse_id: int or str
        load id of subject mouse or 'all' for all mice together
    type: str
        'supervised' or 'unsupervised' or 'raw' - related to class labels to look at
    
    returns
    -------
    
    """
    # loading path on my laptop as default
    path = Path("/Volumes/Lenca_SSD/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)

    # checking that particular mouse_id is a valid option
    if(mouseId!='all'):
        if(int(mouseId) not in subject_ids):
            raise Exception('Mouse id can only be "all" or in ', subject_ids)

    # checking that type of labels is a valid option
    if (type not in ['raw','supervised','unsupervised']):
        raise Exception('Type can only be in', ['raw','supervised','unsupervised'])

    # loading data
    df = load_and_wrangle(mouseId=mouseId, path=path, overwrite=False)

    # info about which class is aggressive behavior
    if(type == 'unsupervised'):
        aggClass = 2
        index_aggClass = 1 # because unsupervised start from 1
    elif(type == 'supervised'):
        aggClass = 3 
        index_aggClass = 3 # because unsupervised start from 0

    # getting possible classes list
    classes = df[f'{type} labels'].unique()
    classes.sort()

    # plotting distribution of classes across all days & all trials
    plt.title("All days")
    plt.bar(classes,df[f'{type} labels'].value_counts()[classes],color='gray')
    plt.bar(aggClass,df[f'{type} labels'].value_counts()[aggClass],color='darkred',label='agg') # coloring aggression red
    plt.ylabel('single frame count')
    plt.xlabel(f'{type} classes')
    plt.xticks(classes)
    plt.legend()
    plt.show()

    # plotting dsitribution of classes for each day
    days = df['day'].unique()
    N = 3
    ind = np.arange(0,len(classes)) 
    width = 0.1
    for day in days:
        temp = df[df['day']==day]
        plt.bar(ind + width*(day-1), temp[f'{type} labels'].value_counts()[classes], width, color=cmapB[day*2],label='day '+str(day))
        plt.bar(ind[index_aggClass] + width*(day-1), temp[f'{type} labels'].value_counts()[aggClass], width, color=cmapR[day*2])
    plt.xlabel(f'{type} classes')
    plt.xticks(ticks = ind + width*4,labels = classes)
    plt.ylabel('single frame count')
    plt.legend()
    plt.show()

    # plotting average latency within each class
    mean = []
    for c in classes:
        condition = np.array(df[f'{type} labels']==c)
        countConsec = np.diff(np.where(np.concatenate(([condition[0]],
                                     condition[:-1] != condition[1:],
                                     [True])))[0])[::2]
        mean.append(countConsec.mean())
    plt.ylabel('mean consecutive frames')
    plt.xlabel(f'{type} classes')
    plt.xticks(classes)
    plt.bar(classes,mean,color='gray')
    plt.bar(aggClass,mean[index_aggClass],color='darkred',label='agg')
    plt.legend()
    plt.show()

    # plotting distribution of aggression latency
    condition = np.array(df[f'{type} labels']==aggClass)
    countConsec = np.diff(np.where(np.concatenate(([condition[0]],
                                     condition[:-1] != condition[1:],
                                     [True])))[0])[::2]
    plt.ylabel('count')
    plt.xlabel('consecutive frames length')
    plt.title('Aggressive Runs')
    plt.hist(countConsec,color='darkred',bins=20)
    plt.show()

    # plotting latency of classes for each day
    for day in days:
        mean = []
        temp = df[df['day']==day]
        for c in classes:
            condition = np.array(temp[f'{type} labels']==c)
            countConsec = np.diff(np.where(np.concatenate(([condition[0]],
                                        condition[:-1] != condition[1:],
                                        [True])))[0])[::2]
            mean.append(countConsec.mean())
        plt.bar(ind + width*(day-1), mean, width, color=cmapB[day*2],label='day '+str(day))
        plt.bar(ind[index_aggClass] + width*(day-1), mean[index_aggClass], width, color=cmapR[day*2])
    plt.ylabel('mean consecutive frames length')
    plt.xlabel(f'{type} classes')
    plt.xticks(ticks = ind + width*4,labels = classes)
    plt.legend()
    plt.show()
        
    # plotting distribution of classes preceding aggression
    indicesAgg = np.array(df[df[f'{type} labels']==aggClass].index)
    indicesPre = indicesAgg - 1
    temp = df.iloc[indicesPre,:]
    plt.title("Class preceeding aggression start")
    plt.bar(classes[:index_aggClass],temp[f'{type} labels'].value_counts()[classes[:index_aggClass]],color='gray')
    plt.bar(classes[index_aggClass+1:],temp[f'{type} labels'].value_counts()[classes[index_aggClass+1:]],color='gray')
    plt.ylabel('count')
    plt.xlabel(f'{type} classes')
    plt.xticks(classes)
    plt.show()


# function to plot histograms of features values for an animal
def histogram_feature(animal, group, features, ind_day=8, path=None):
    
    path = Path("/Volumes/Lenca_SSD/github/Falkner_Multi-region_Aggression/data") if path is None else Path(path)
    
    df = load_and_wrangle(mouseId=animal, group=group, path=path, overwrite=False)
    days = np.unique(df['day'])
    trials = np.unique(df['trial'])

    X = np.empty((len(days)), dtype=object)
    for ind_feature in range(len(features)):
        a = np.empty((len(trials)), dtype=object) # all features across sessions to get optimal bin partition
        c = 0 # counting index
        for ind_trial in range(0,len(trials)): # trial index
            temp = df[df['day'] == days[ind_day]]
            temp = temp[temp['trial'] == trials[ind_trial]].reset_index()
            behav = pd.read_parquet(f'../data/processed_features_020924_parquets/{animal}_{days[ind_day]}_{temp.loc[0,"other"]}_{trials[ind_trial]}_zscored_features.parquet')
            a[c] = np.array(behav[features[ind_feature]])
            c = c + 1
            

        all = np.concatenate(a, axis=0)

        bin_edges = np.histogram_bin_edges(all, bins='fd') # optimal number of bins with fd method
        # _, bin_edges = np.histogram(all, bins=Nbins)
        plt.figure()
        plt.title(f' {animal} ({group}) - Day {ind_day+1}')
        plt.xlabel(features[ind_feature] + ' histogram')
        plt.hist(all, bins=bin_edges)
        plt.show()

