# importing modules and packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from io_utils import *
from pathlib import Path
import matplotlib as mpl

cmapB = mpl.cm.Blues(np.linspace(0,1,20))
cmapR = mpl.cm.Reds(np.linspace(0,1,20))
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
    # checking that particular mouse_id is a valid option
    if (int(mouseId) not in subject_ids and mouseId!='all'):
        raise Exception('Mouse id can only be "all" or in ', subject_ids)

    # checking that type of labels is a valid option
    if (type not in ['raw','supervised','unsupervised']):
        raise Exception('Type can only be in', ['raw','supervised','unsupervised'])

    # loading data
    df = load_and_wrangle(mouseId=mouseId, path=None, overwrite=False)

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
