# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def one_hot_encoder(Y):
    #gets a categorical variable, returns the one_hot encoded and the classes
    Y_set = np.sort(np.asarray(list(set(Y))))
    K = len(Y_set)
    Y_dic = dict()
    for i, item in enumerate(Y_set):
        Y_dic[item] = i
    N = len(Y)
    Y_enc = np.zeros((N,K))
    for i in range(N):
        Y_enc[i, Y_dic[Y[i]]] = 1
    Y_enc = Y_enc.astype('int32')
    return Y_enc, Y_dic.keys()

def get_normalized_data(s): #s = 'train' or 'test'
    # opens data corresonding to kaggle's digit recognition challenge
    # https://www.kaggle.com/c/digit-recognizer/
    # train data has 785 columns (28*28)+ 1 label
    # test data has 784 colums 
    path_dir = 'C:\\Users\\digits\\' 
    if s == 'train':
        df = pd.read_csv(path_dir+s+'.csv')
        data = df.values
        np.random.shuffle(data)
        Y = data[:,0]
        X = data[:,1:].astype(np.float32)
        Xmean = np.average(X, axis=0)
        Xstd = np.std(X, axis=0)
        np.place(Xstd, Xstd == 0, 1)
        X = (X - Xmean.T) / Xstd
        return X, Y
    elif s == 'test':
        df = pd.read_csv(path_dir+s+'.csv')
        X = df.values.astype(np.float32)
        np.random.shuffle(X)
        Xmean = np.average(X, axis=0)
        Xstd = np.std(X, axis=0)
        np.place(Xstd, Xstd == 0, 1)
        X = (X - Xmean.T) / Xstd
        return X, None
    else:
        print("File doesn't exist!")
        return None, None
        

