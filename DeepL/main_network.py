import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
#include utilities
from utilities import *
from main_neuron import flatten



def Initialisation(n0,n1,n2):
    w1 = np.random.randn(n1,n0)
    b1 = np.random.randn(n1, 1)
    w2 = np.random.randn(n2,n1)
    b2 = np.random.randn(n2, 1)
    
    parameter = {'w1' : w1,
                 'b1' : b1,
                 'w2' : w2,
                 'b2' : b2
                 }
    
    return parameter


def Forward_propagation(X,parameter):
    w1 = parameter['w1']
    b1 = parameter['b1']
    w2 = parameter['w2']
    b2 = parameter['b2']

    Z1 = w1.dot(X) + b1
    A1 = 1/(1+ np.exp(-Z1))
    Z2 = w2.dot(A1) + b2
    A2 = 1/(1+ np.exp(-Z2))

    activation ={
        'A1' : A1,
        'A2' : A2
    }
    return activation

#A = Model(X,W,b)

def Cost(A,y):
    return -1/len(y)*np.sum(y*np.log(A) + (1-y)*np.log(1-A))
#L = Cost(A,y)

def Back_propagation(X, y,activation,parameter):
    A1 = activation['A1']
    A2 = activation['A2']
    w2 = parameter['w2']
    m = y.shape[1]

    dZ2 = A2 -y
    dw2= 1/m*dZ2.dot(A1)
    db2 = 1/m*np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(w2.T, dZ2)*A1*(1-A1)
    dw1 = 1/m*dZ1.dot(X.T)
    db1 = 1/m*np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        'dW1' : dw1 ,
        'db1' : db1,
        'dW2' : dw2,
        'db2' : db2
    }

    return gradients
#dW,db = Gradients(A,X,y)

def Update(parameter, gradients,al): #al(pha) is the learning rate
    w1 = parameter['w1']
    b1 = parameter['b1']
    w2 = parameter['w2']
    b2 = parameter['b2']

    dw1 = gradients['dw1']
    dw2 = gradients['dw2']
    db1 = gradients['db1']
    db2 = gradients['db2']

    w1 = w1  - al*dw1
    b1 = b1 -al*db1

    w2 = w2 - al*dw2
    b2 = b2 -al*db2

    parameter['w1'] = w1
    parameter['b1'] = b1
    parameter['w2'] = w2
    parameter['b2'] = b2
    
    return parameter 

#predictions for future date
def predict(X,parameter):
    activation = Forward_propagation(X, parameter)
    A2 = activation['A2']
    return A2 >= 0.5

def Neural_network(X,y,n1, a=1, n_iter=100):
    n0 = X.shape[0]
    n2 = y.shape[0]
    parameter = Initialisation(n0,n1,n2)

    Loss = [] #to gather the data from the function Cost
    Acc = []
    
    for i in range(n_iter):
        activation = Forward_propagation(X,parameter)
        gradients = Back_propagation(X,y,activation,parameter)
        parameter = Update(parameter,gradients,a)

        if i % 10 ==0 :
             Loss.append(Cost(activation['A2'],y))
             y_pred = predict(X,parameter)
             curr_accuracy = accuracy_score(flatten(y),flatten(y_pred))
             Acc.append(curr_accuracy)
        
    plt.figure(figsize= (14,4))

    plt.subplot(1,2,1)
    plt.plot(Loss, label = 'Train loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(Acc, label = 'Train acc')
    plt.legend()
    plt.show()

    return parameter




#Create my neural network:
#global neuron network:
def Initialisation_Global(*args):
    W = [ "w{}".format(i + 1) for i in range(len(args) - 1) ]
    b = [ "b{}".format(i + 1) for i in range(len(args) - 1) ]

    parametres = {}

    for i in range(len(args)):
        parametres[W[i]] = np.random.randn(args[i + 1],args[i])
        parametres[b[i]] = np.random.randn(args[i + 1],1)

    return parametres

def FP_global(X,parameter,args):
    
    activations = {}
    neuron = {}

    W = [ "w{}".format(i+1) for i in range(len(args)) ]
    b = [ "b{}".format(i+1) for i in range(len(args)) ]
    Z = [ "Z{}".format(i+1) for i in range(len(args)) ]
    A = [ "A{}".format(i+1) for i in range(len(args)) ]
    for i in range(len(args) - 1):
        W[i] = parameter[ W[i] ]
        b[i] = parameter[ b[i] ]

        if i == 0 : 
            neuron[Z[i]] = W[i].dot(X) + b[i]
        else : 
            neuron[Z[i]] = W[i].dot(activations[A[i]]) + b[i]

        activations[A[i]] = 1/(1+ np.exp(-neuron[Z[i]]))
    
    return activations

def BP_global(X, y, activation, parameter, args):
    gradients = {}

    L = len(args) - 1  # Number of layers
    m = y.shape[1]

    W = ["w{}".format(i + 1) for i in range(L)]
    b = ["b{}".format(i + 1) for i in range(L)]
    A = ["A{}".format(i + 1) for i in range(L)]

    # Initialize gradient dictionaries
    dW = {}
    db = {}
    dA = {}
    dZ = {}

    # Compute the gradient for the last layer
    dZ[L - 1] = activation[A[L - 1]] - y
    dW[L - 1] = 1 / m * np.dot(dZ[L - 1], activation[A[L - 2]].T if L > 1 else X.T)
    db[L - 1] = 1 / m * np.sum(dZ[L - 1], axis=1, keepdims=True)

    for i in reversed(range(L - 1)):
        dA[i] = np.dot(parameter[W[i + 1]].T, dZ[i + 1])
        dZ[i] = dA[i] * activation[A[i]] * (1 - activation[A[i]])
        dW[i] = 1 / m * np.dot(dZ[i], activation[A[i - 1]].T if i > 0 else X.T)
        db[i] = 1 / m * np.sum(dZ[i], axis=1, keepdims=True)

    for i in range(L):
        gradients['d' + W[i]] = dW[i]
        gradients['d' + b[i]] = db[i]

    return gradients

def Update_global(parameter, gradients,al,args):
    W = [ "w{}".format(i+1) for i in range(len(args)) ]
    b = [ "b{}".format(i+1) for i in range(len(args)) ]
    dW = [ "dw{}".format(i+1) for i in range(len(args)) ]
    db = [ "db{}".format(i+1) for i in range(len(args)) ]

    for i in range(len(args) - 1):
        parameter[W[i]] = parameter[W[i]] - al * gradients[dW[i]]
        parameter[b[i]] = parameter[b[i]] - al * gradients[db[i]]

    return parameter

def predict_global(X, parameter, args):
    activation = FP_global(X,parameter,args)
    A = [ "A{}".format(i+1) for i in range(len(args)) ]
    return activation[A[-1]] >= 0.5

def Neural_network_global(X,y,al=1, n_iter = 100, *args ):
    if args[0] != X.shape[0] or args[-1] != y.shape[0]:return None
    else:
        parameter = Initialisation_Global(*args)

        Loss = []
        Acc = []

        for i in range(n_iter):
            activation = FP_global(X,parameter, args)
            gradients = BP_global(X, y,activation,parameter,args)
            parameter = Update_global(parameter,gradients,al,args)

            if i % 10 ==0:
                A = [ "A{}".format(j+1) for j in range(len(args)) ]
                Loss.append(Cost(activation[A[-1]],y))
                y_pred = predict_global(X,parameter,args)
                curr_accuracy = accuracy_score(flatten(y),flatten(y_pred))
                Acc.append(curr_accuracy)
        plt.figure(figsize= (14,4))

        plt.subplot(1,2,1)
        plt.plot(Loss, label = 'Train loss')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(Acc, label = 'Train acc')
        plt.legend()
        plt.show()

        return parameter

def Neural_network_global1(X, y, al=1, n_iter=100, *args):
    if args[0] != X.shape[0] or args[-1] != y.shape[0]:
        return None
    else:
        parameter = Initialisation_Global(*args)
        Loss = []
        Acc = []

        for j in range(n_iter):
            activation = FP_global(X, parameter, args)
            gradients = BP_global(X, y, activation, parameter, args)
            parameter = Update_global(parameter, gradients, al, args)

            if j % 10 == 0:
                A = ["A{}".format(i + 1) for i in range(len(args) - 1)]
                Loss.append(Cost(activation[A[-1]], y))
                y_pred = predict(X, parameter, args)
                curr_accuracy = accuracy_score(flatten(y), flatten(y_pred))
                Acc.append(curr_accuracy)

        plt.figure(figsize=(14, 4))

        plt.subplot(1, 2, 1)
        plt.plot(Loss, label='Train loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(Acc, label='Train acc')
        plt.legend()
        plt.show()

        return parameter