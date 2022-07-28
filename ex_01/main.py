import numpy as np
from sklearn import datasets
import pandas as pd

if __name__ == '__main__':
    print("hello dai ca Linh")

    input_matrix = np.random.randint(0, 100, (10, 3))
    print(input_matrix)

    kernel_matrix = np.dot(input_matrix, input_matrix.T)
    print(f'kernel matrix: {kernel_matrix}')