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
    return 1/Z.shape[0] * np.dot(Z.T, Z)


def choose_dim(evls, alpha):
    s = np.sum(evls)
    d = len(evls)
    for r in range(d):
        if np.sum(evls[0:r]) / s > alpha:
            return r
    return d


def pca(X, alpha):
    mu = mean(X)
    Z = center_data(X, mu)
    sigma = covariance_matrix(Z)
    evls, evts = eigen(sigma)
    r = choose_dim(evls, alpha)
    Ur = evts[0:r, :]
    return np.dot()



