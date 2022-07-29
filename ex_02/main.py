import numpy as np
from numpy.linalg import eig
import component as cp

np.random.seed(2)

if __name__ == '__main__':
    print(f'hello dai ca Linh')

    A = np.array([[0.681, -0.039, 1.265],
                  [-0.039, 0.187, -0.32],
                  [1.265, -0.32, 3.093]])

    evls, evts = cp.eigen(A)

    print(f'eigenvalues: {type(evls)}, evls: {evls}')
    print(f'eigenvectors: {evts}')

    print(f'mean vector: {cp.mean(A)}')
