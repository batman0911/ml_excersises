import numpy as np
from sklearn import datasets

import component as cp
import ex_set_01.ex_01.kernel as kn

np.random.seed(2)

if __name__ == '__main__':

    iris = datasets.load_iris()

    X = iris.data

    print(f'shape of iris: {X.shape}')

    print('\n\n-------- pca -----------')
    reduce_X = cp.pca(X, alpha=0.99)
    print(f'reduce dim with pca: \n {reduce_X}')
    print('-------- pca -----------')

    print('\n\n-------- kernel pca ----')
    K_identity = kn.identity_kernel(X)
    print(f'identity kernel: \n {K_identity}')
    K_gaussian = kn.gaussian_kernel(X, 1)
    print(f'gaussian kernel: \n {K_gaussian}')
    reduce_X_kn_cpa = cp.kernel_pca(K_gaussian, X.shape[1], 0.99)
    print(f'reduce dim with kernel cpa: \n {reduce_X_kn_cpa}')
    print('-------- kernel pca ----')

    print('\n\n-------- svd -----------')
    X = cp.center_data(X, cp.mean(X))
    u, s, vh = cp.svd(X)
    print(f'singular value: {s}')

    # check with np
    un, sn,  vhn = np.linalg.svd(X)
    print(f'singular values by np: {sn}')
    print('-------- svd -----------')