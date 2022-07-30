import numpy as np

import kernel as kn

np.set_printoptions(precision=3)

np.random.seed(100)

if __name__ == '__main__':

    input_matrix = np.random.randint(0, 100, (10, 3))
    print(input_matrix)

    K_identity = kn.identity_kernel(input_matrix)
    print(f'identity kernel matrix: \n {K_identity}')

    K_gaussian = kn.gaussian_kernel(input_matrix, 150)
    print(f'gaussian kernel matrix: \n {K_gaussian}')