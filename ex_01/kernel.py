import numpy as np


class Kernel:
    def __int__(self,
                kernel_type,
                input_matrix):
        self.kernel_type = kernel_type
        self.input_matrix = input_matrix

    def kernel(self):
        pass

    @classmethod
    def identity_kernel(cls, X):
        print(f'return the identity kernel matrix')
        return np.dot(X, X.T)

    @classmethod
    def gaussian_kernel(cls, X):
        print(f'return the gaussian kernel matrix')
        pass

    @classmethod
    def kij_gaussian(cls, x, y, sigma):
        return np.exp(- 0.5 * np.dot(x - y, (x - y).T) / sigma**2)
