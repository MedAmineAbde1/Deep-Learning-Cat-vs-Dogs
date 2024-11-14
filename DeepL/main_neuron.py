import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
#include utilities
from utilities import *


#the Implimentation of an artificial neuron

#the components of the AN
def Initialisation(X):
    W = np.random.randn(X.shape[1],1)
    b = np.random.randn(1,1)[0]
    return W,b

def Model(X,W,b):
    Z = X.dot(W) + b
    A = 1/(1+ np.exp(-Z))
    return A

def Cost(A,y):
    return -1/len(y)*sum(y*np.log(A) + (1-y)*np.log(1-A))

def Gradients(A,X,y):
    dW = 1/len(y)*np.dot(X.T,A-y)
    db = 1/len(y)*np.sum(A-y)
    return dW,db

def Update(W,dW,b,db,a = 1):
    W = W - a*dW
    b = b - a*db
    return W,b

def prediction(X,W,b):
    A = Model(X,W,b)
    return A >= 0.5

#The Artificial Neuron

def Artificial_Neuron(X,y, a=1, n_iter=100,show = 'l'):
    W,b = Initialisation(X)
    Loss = [] #to gather the data from the function Cost
    Acc = []
    #Training the Artificial Neuron
    for i in range(n_iter):
        A = Model(X,W,b)
        y_pred = prediction(X,W,b)
        Acc.append(accuracy_score(y,y_pred))
        Loss.append(Cost(A,y))
        dW,db = Gradients(A,X,y)
        W,b = Update(W,dW,b,db,a)
        
    if show == 'a':
        plt.plot(Acc)
    else:
        plt.plot(Loss)
    plt.show()
    return W,b

X_train, y_train, X_test, y_test = load_data()  #fetch die datasets aus dem File.


# 1. Normaliser le train_set et le test_set (0-255 -> 0-1) -> Success
def normalise_1(d_set:np):
    u = np.linspace(0,1,256)
    d = d_set.ravel()
    #for i in range(d_set.size):
     #   d[i] = u[d[i]]
    #return d.reshape(d_set.shape)
    return u[d].reshape(d_set.shape)

def normalise(d_set:np): return d_set / 255.0 
#for i in range(d.shape[0]):print(d[i,:,:] , "für i=" + str(i))


# 2. flatten() les variables du train_set et du test_set (64x64 -> 4096) -> Success
def flatten(d:np): #erste prototype
    l=[np.ones((2,2)) for i in range(d.shape[0])]
    for i in range(d.shape[0]):
        l[i] = d[i,:,:] 
        l[i] = d[i].ravel()
    return np.array(l) 
def flatten1(d:np): #way more efficient
    return d.reshape(d.shape[0],d.shape[1]*d.shape[2])

# 3. Entrainer le modele sur le train_set (tracer la courbe d'apprentissage, trouver les bons hyper-params) -> Success
neo_x = X_train
for i in range(neo_x.shape[0]):
    neo_x[i] = normalise(neo_x[i])

neo_x = flatten(neo_x)

#W,b = Artificial_Neuron(neo_x,y_train,show='l')

# 4. Évaluer le modele sur le test_set (tracer également la courbe de Loss pour le test_set)
neo = X_test
for i in range(neo.shape[0]):
    neo[i] = normalise(neo[i])

neo = flatten(neo)
W,b = Artificial_Neuron(neo,y_test)

print(f'das ist die werte von W:{W}\n')
print(f'das ist die werte von b:{b}\n')
