import numpy as np
from sklearn import datasets
import pandas as pd
from kernel import Kernel

# np.set_printoptions(precision=10)

np.random.seed(100)

if __name__ == '__main__':
    print("hello dai ca Linh")

    input_matrix = np.random.randint(0, 100, (10, 3))
    print(input_matrix)

    kernel = Kernel()
    K_identity = kernel.identity_kernel(input_matrix)
    print(f'identity kernel matrix: \n {K_identity}')

    K_gaussian = kernel.gaussian_kernel(input_matrix)
    print(f'gaussian kernel matrix: \n {K_gaussian}')