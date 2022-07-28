import numpy as np


class Kernel:
    def __int__(self,
                kernel_type=None,
                input_matrix=None):
        self.kernel_type = kernel_type
        self.input_matrix = input_matrix

    def identity_kernel(self, X):
        print(f'return the identity kernel matrix')
        return np.dot(X, X.T)

    def gaussian_kernel(self, X):
        print(f'return the gaussian kernel matrix')
        N = X.shape[0]
        sigma = 5 * np.sqrt(6)
        K = np.zeros((N, N))
        tol = 1e-6
        for i in range(N):
            for j in range(N):
                k = self.kij_gaussian(X[i, :], X[j, :], sigma)
                if abs(k) < tol:
                    k = 0
                K[i, j] = K[j, i] = np.around(k, decimals=4)
        return K

    @classmethod
    def kij_gaussian(cls, x, y, sigma):
        return np.exp(- 0.5 * np.dot(x - y, (x - y).T) / sigma ** 2)
