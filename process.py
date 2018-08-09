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
    


def get_face_Data(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    # opens data corresonding to kaggle's Facial Expression Recognition Challenge
    # https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/
    # data has 3 columns, first is label with 7 states
    # second is 2304 pixel data separated with spaces
    # third says whether the row is training or test
    data_path = "C:\\Users\\Payam\\Dropbox\\LazyProgrammerCourses-Mycodes\\ANN\\fer2013\\"
    Y = []
    X = []
    first = True
    for line in open(data_path+'fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y
        
        

