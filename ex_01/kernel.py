import numpy as np


def identity_kernel(X):
    print(f'return the identity kernel matrix')
    return np.dot(X, X.T)


def kij_gaussian(x, y, sigma2):
    return np.exp(- 0.5 * np.dot(x - y, (x - y).T) / sigma2)


def gaussian_kernel(X, sigma2=1):
    print(f'return the gaussian kernel matrix')
    N = X.shape[0]
    K = np.zeros((N, N))
    tol = 1e-6
    for i in range(N):
        for j in range(N):
            k = kij_gaussian(X[i, :], X[j, :], sigma2)
            if abs(k) < tol:
                k = 0
            K[i, j] = K[j, i] = np.around(k, decimals=4)
    return K


class Kernel:
    def __int__(self,
                kernel_type=None,
                input_matrix=None):
        self.kernel_type = kernel_type
        self.input_matrix = input_matrix
