import numpy as np
import numpy.linalg as linalg


def mean(X):
    return 1 / X.shape[0] * np.sum(X, axis=0)


def eigen(A):
    eigen_values, eigen_vectors = linalg.eig(A)

    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    return eigen_values, eigen_vectors


def center_data(X, mean):
    Z = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        Z[i, :] = X[i, :] - mean
    return Z


def covariance_matrix(Z):
    1/Z.shape[0] * np.dot(Z.T, Z)


def fraction_variance(evls, r):
    return np.sum(evls[0:r]) / np.sum(evls)
