# -*- coding: utf-8 -*-

import numpy as np
          
def sigmoid(a):
    return 1. / (1. + np.exp(-a))

def sigmoid_grad(a):
    s = sigmoid(a)
    ds = (1-s)*s
    return ds

def softmax(A):
    expA = np.exp(A)
    expA = expA / np.sum(expA, axis = 1, keepdims=True)
    return expA

def cost(T, Y): #T (one_hot_encoded) (N*K),  Y (probability prediction of each class) (N*K)
    return np.sum(-T * np.log(Y))

def mean_cost(Y, Y_hat): # Y (N*1) is NOT hot encoded, Y_hat (N*K) is probability prediction for each class
    N = len(Y)
    #print(Y.dtype)
    J_mean = np.average( -np.log(Y_hat[np.arange(N), Y]) )
    return J_mean

def classification_rate(Y_pred, Y): #returns num_correct/ num_all, Y and Y_pred are single column, containing the index of the class. They are NOT one_hot_encoded
    return float(np.sum(Y_pred == Y)) / len(Y)


# sigmoid derivatives
    
# in this case i is the randomly chosen data point that must be passed along to all the gradients
def der_l2_sigmoid(T, Y, Z1, i = -1): #derivatives in the second (last) layer
    if i == -1:
        Y_min_T = Y-T
        db2 = np.sum(Y_min_T, axis=0) # (y-t) summed over all data points
        dW2 =  np.transpose(sigmoid(Z1)).dot(Y_min_T) # summed over all data points
    # for stochastic gradient descent:
    else:
        Y_min_T = Y[i] - T[i]
        db2 = Y_min_T
        dW2 = np.outer(sigmoid(Z1[i]), Y_min_T)
    return db2, dW2

def der_l1_sigmoid(T, Y, Z1, W2, X, i = -1):  #derivatives in the first (closer to input) layer
    if i == -1:
        dZ = (Y-T).dot(W2.T) * sigmoid_grad(Z1) 
        db1 = np.sum(dZ, axis=0) # summed over all data points
        dW1 = np.transpose(X).dot(dZ) # summed over all data points
    # for stochastic gradient descent:
    else:
        dZ = (Y[i]-T[i]).dot(W2.T) * sigmoid_grad(Z1[i])
        db1 = dZ
        dW1 = np.outer( X[i], dZ )
    return db1, dW1

# tanh derivatives
    
# in this case i is the randomly chosen data point that must be passed along to all the gradients
def der_l2_tanh(T, Y, Z1, i = -1): #derivatives in the second (last) layer
    if i == -1:
        Y_min_T = Y-T
        db2 = np.sum(Y_min_T, axis=0) # (y-t) summed over all data points
        dW2 =  np.transpose(np.tanh(Z1)).dot(Y_min_T) # summed over all data points
    # for stochastic gradient descent:
    else:
        Y_min_T = Y[i] - T[i]
        db2 = Y_min_T
        dW2 = np.outer(np.tanh(Z1[i]), Y_min_T)
    return db2, dW2
    
def der_l1_tanh(T, Y, Z1, W2, X, i = -1):  #derivatives in the first (closer to input) layer
    if i == -1:
        dZ = (Y-T).dot(W2.T) * (1 -Z1*Z1) 
        db1 = np.sum(dZ, axis=0) # summed over all data points
        dW1 = np.transpose(X).dot(dZ) # summed over all data points
    # for stochastic gradient descent:
    else:
        dZ = (Y[i]-T[i]).dot(W2.T) * (1- Z1[i]*Z1[i])
        db1 = dZ
        dW1 = np.outer( X[i], dZ )
    return db1, dW1

# relu derivatives
    
# in this case i is the randomly chosen data point that must be passed along to all the gradients
def der_l2_relu(T, Y, Z1, i = -1): #derivatives in the second (last) layer
    if i == -1:
        Y_min_T = Y-T
        db2 = np.sum(Y_min_T, axis=0) # (y-t) summed over all data points
        dW2 =  np.transpose(np.maximum(0, Z1)).dot(Y_min_T) # summed over all data points
    # for stochastic gradient descent:
    else:
        Y_min_T = Y[i] - T[i]
        db2 = Y_min_T
        dW2 = np.outer(np.maximum(0, Z1[i]), Y_min_T)
    return db2, dW2
    
def der_l1_relu(T, Y, Z1, W2, X, i = -1):  #derivatives in the first (closer to input) layer
    if i == -1:
        dZ = (Y-T).dot(W2.T) * (Z1 > 0) 
        db1 = np.sum(dZ, axis=0) # summed over all data points
        dW1 = np.transpose(X).dot(dZ) # summed over all data points
    # for stochastic gradient descent:
    else:
        dZ = (Y[i]-T[i]).dot(W2.T) * (Z1[i] > 0)
        db1 = dZ
        dW1 = np.outer( X[i], dZ )
    return db1, dW1
