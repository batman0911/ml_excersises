import numpy as np
from sklearn import datasets
import pandas as pd
from kernel import Kernel

if __name__ == '__main__':
    print("hello dai ca Linh")

    input_matrix = np.random.randint(0, 100, (10, 3))
    print(input_matrix)

    kernel = Kernel()
    K = kernel.identity_kernel(input_matrix)
    print(f'identity kernel matrix: \n {K}')
