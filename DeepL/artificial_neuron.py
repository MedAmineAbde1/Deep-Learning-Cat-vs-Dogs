
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import h5py
from utilities import *

# Generate synthetic data
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape(-1, 1)

#print('Dimension of X:', X.shape)
#print('Dimension of y:', y.shape)

# Plot the data points
#plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='summer')
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.title('Scatter plot of the generated blobs')
#plt.show()

X_train, y_train, X_test, y_test = load_data()
print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))

#plt.figure(figsize=(16, 8))
#for i in range(1, 10):
#    plt.subplot(4, 5, i)
#    plt.imshow(X_train[i], cmap='gray')
#    plt.title(y_train[i])
#    plt.tight_layout()
#plt.show()