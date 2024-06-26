import argparse
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define numpy arrays
A = np.random.rand(3, 5)
B = np.zeros((3, 1))
C = np.ones((1, 3))
D = np.linspace(0, 5, num=6)
E = np.array([[1,2,3,4],[5,6,7,8]])

D = E.shape[0] # feature dimension
N = E.shape[1] # number of data instances

print(D)
print(N)

# print(E)
# print(E.ndim)
# print(E.shape)
# print(np.mean(E, axis = 1))
# print(np.mean(E, axis = 1).shape)
# print((np.mean(E, axis = 1)).reshape(2,1))

# mu = (np.mean(E, axis = 1)).reshape(2,1)
# print(mu)
# print(E - mu)

# print(E ** 2)
# print(np.sum(E, axis = 0))
# print((np.sum(E, axis = 0) / 3))

mu = np.mean(E, axis = 1).reshape(D, 1)
print("E:")
print(E)
print("mu:")
print(mu)

workingS = E - mu # broadcasting mu vector to subtract each column vector in workingS
workingS = workingS ** 2 # squares each number in the matrix
print("Squared data:")
print(workingS)
sumOfRows = np.sum(workingS, axis = 1) # sums the rows of workingS
print("sumOfRows:")
print(sumOfRows)
varianceArray = (sumOfRows / (N - 1)).reshape(D, 1) # gives the variance for each feature in a D-by-1 matrix

print("Variance: ")
print(varianceArray)

print("real cov of E:")
print(np.cov(E))
print()

X = np.copy(E)
mean_X = np.mean(X, axis=1, keepdims=True)
print("Mean with keepdims:")
print(mean_X)

# Subtract the mean from each element to get a zero-mean matrix
X_zero_mean = X - mean_X

# Calculate the covariance matrix
cov_matrix = (X_zero_mean @ X_zero_mean.T) / (X.shape[1] - 1)

print(cov_matrix)

X = np.array([[20, 8, -6, 6], [5, -2, -3, 4]])

print(np.cov(X))

negativeSquareRootC = np.array([[0.133,0.096],[0.096,0.418]])
z = np.array([-1,-3])

print()
print(np.matmul(negativeSquareRootC, z))
"""
# Print
print(A)
print(B)
print(C)
print(D)
print(type(D))

# Print shapes
print(A.shape)
print(B.shape)
print(C.shape)
print(D.shape)

print(A.ndim)
print(B.ndim)
print(C.ndim)
print(D.ndim)

# Reshape
print(D.reshape(6, 1)) # 6-by-1 matrix (column vector)
print(D.reshape(6, 1).shape)
print(D.reshape(6, 1).ndim)

print(D.reshape(1, 6)) # 1-by-6 matrix (row vector)
print(D.reshape(1, 6).shape)
print(D.reshape(1, 6).ndim)

print(D.reshape(-1, 1))
print(D.reshape(-1, 1).shape)
print(D.reshape(-1, 1).ndim)


# Extract element (note that, numpy index starts from 0)
print(A)
print(A[1,1])
print(A[:, 1])
print(A[1, :])
print(A[1, -1])
print(A[1, 0:3])

# matrix transpose
print(A)
print(A.transpose())

# real values
print(-4)
print((-4)**0.5)
print(np.real(-4**0.5))

# matrix multiplication
print(np.matmul(A.transpose(), A))
print(np.matmul(A, A.transpose()))
print(np.matmul(B, C))
print(np.matmul(C, B))

# matrix inversion
print(np.linalg.inv(np.matmul(A, A.transpose())))

# eigendecomposition
Sigma = np.matmul(A, A.transpose())
V, W = np.linalg.eig(Sigma)
print(V)
print(np.argsort(V))
print(np.argsort(V)[::-1])
print(V[np.argsort(V)])
"""