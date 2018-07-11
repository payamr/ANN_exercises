# -*- coding: utf-8 -*-

import numpy as np
import numpyam as npm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def one_hot_encoder(Y):
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

def test_train_split(X, Y, ratio = 0.2):
    N = len(X)
    N2 = len(Y)
    if N != N2:
        print('this data cannot be split!')
        return None
    X, Y = shuffle(X, Y)
    N_test = int(N*ratio)
    X_test, Y_test = X[:N_test], Y[:N_test]
    X_train, Y_train = X[N_test:], Y[N_test:]
    return X_train, X_test, Y_train, Y_test
      
 
class ANN(object):
    def __init__(self, M, activation='relu'):
        self.M = M
        self.activation = activation
    
    def forward_sigmoid(self, X):
        Z1 = X.dot(self.W1) + self.b1
        H = npm.sigmoid(Z1) 
        theta = H.dot(self.W2) + self.b2
        Y_hat = npm.softmax(theta)
        return Y_hat, Z1

    def forward_tanh(self, X):
        Z1 = X.dot(self.W1) + self.b1
        H = np.tanh(Z1)
        theta = H.dot(self.W2) + self.b2
        Y_hat = npm.softmax(theta)
        return Y_hat, Z1

    def forward_relu(self, X):
        Z1 = X.dot(self.W1) + self.b1
        H = np.maximum(0, Z1) #relu
        theta = H.dot(self.W2) + self.b2
        Y_hat = npm.softmax(theta)
        return Y_hat, Z1

    def fit(self, X, Y, learning_rate = 1e-6, regular = 1e-4, epoch_num=1000):
        X, Xvalid, Y, Yvalid = test_train_split(X, Y)
        N, D = X.shape
        T, self.classes = one_hot_encoder(Y)
        Tvalid, _ = one_hot_encoder(Yvalid)
        K = T.shape[1]
        
        self.forward = getattr(self, 'forward_'+self.activation)
        der_l1 = getattr(npm, 'der_l1_'+self.activation)
        der_l2 = getattr(npm, 'der_l2_'+self.activation)

        self.b1 = np.random.randn(self.M)
        self.W1 = np.random.randn(D,self.M)
        self.b2 = np.random.randn(K)
        self.W2 = np.random.randn(self.M,K)
        
        self.cost_train = [] # list of training cost functions
        self.cost_test = [] # list of cost functions for test set
        for epoch in range(epoch_num):
            Y_hat, Z1 = self.forward(X)
            der_b1, der_W1 = der_l1(T, Y_hat, Z1, self.W2, X)
            der_b2, der_W2 = der_l2(T, Y_hat, Z1)
            self.b1 -= learning_rate * ( der_b1 + regular*self.b1 )
            self.W1 -= learning_rate * ( der_W1 + regular*self.W1 )
            self.b2 -= learning_rate * ( der_b2 + regular*self.b2 )
            self.W2 -= learning_rate * ( der_W2 + regular*self.W2 )
            
            J = npm.mean_cost(Y, Y_hat)
            self.cost_train.append(J)
            
            Y_hat_valid, _ = self.forward(Xvalid)
            Jvalid = npm.mean_cost(Yvalid, Y_hat_valid)
            self.cost_test.append(Jvalid)
            
    def predict(self, X):
        Y_hat, _ = self.forward(X)
        return np.argmax(Y_hat, axis=1)
    
    def classification_rate(self, X, Y): #returns num_correct/ num_all
        Y_pred = self.predict(X)
        return float(np.sum(Y_pred == Y)) / len(Y)

