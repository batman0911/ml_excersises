import numpy as np
from sklearn import datasets

import component as cp

np.random.seed(2)

if __name__ == '__main__':

    iris = datasets.load_iris()

    X = iris.data

    print(f'shape of iris: {X.shape}')

    reduce_X = cp.pca(X, alpha=0.95)
    print(f'reduce dim with pca: \n {reduce_X}')

    X = cp.center_data(X, cp.mean(X))
    u, s, vh = cp.svd(X)
    print(f'singular value: {s}')

    # check with np
    un, sn,  vhn = np.linalg.svd(X)
    print(f'singular values by np: {sn}')