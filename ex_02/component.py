import numpy as np
import numpy.linalg as linalg
from sklearn.decomposition import PCA


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
    return np.dot(X, Ur.T)


def svd(A):
    sigma2, VT = eigen(np.dot(A.T, A))
    sigma = np.sqrt(sigma2)
    U = np.dot(A, VT.T)
    for j in range(U.shape[1]):
        U[:, j] = U[:, j] / sigma[j]
    return U, sigma, VT



if __name__ == '__main__':
    X = np.array([
        [1, 2, 1],
        [1, 1, 0],
        [2, 1, 1],
        [3, 4, 1],
        [0, 1, 2]
    ])

    # A = pca(X, 0.9)
    # print(A)
    #
    # pcask = PCA(n_components=2)
    # fit = pcask.fit_transform(X)
    # print(fit)

    u, s, vh = np.linalg.svd(X)
    u1, s1, vh1 = svd(X)

    print(f'u: {u} \n\n s: {s} \n\n vh: {vh}')
    print(f'\n\n')
    print(f'u: {u1} \n\n s: {s1} \n\n vh: {vh1}')

    A = np.dot(u1, np.diag(s1))
    A = np.dot(A, vh1)
    print(f'\n\n A: {A}')

