"""Each party preprocess locally
Aggregrate in the cipher-platform.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from jacobi import *


def mult_block(a, k=3):
    a_list = []
    leng, dim = a.shape

    len_bin = leng // 3
    last = leng - len_bin*k

    a_list = [a[i*len_bin:(i+1)*len_bin] for i in range(k)]
    a_list.append(a[-last:])

    block = np.zeros((dim, dim))
    for i in range(len(a_list)):
        block += np.dot(a_list[i].T, a_list[i])

    np.testing.assert_almost_equal(block, np.dot(a.T, a))
    
    return block


def min_max_scaler(X):
    """Using min max scaler for data standardization then rescale them to 0-5.
    """
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return X

def standard(X):
    """Standard scaler for data
    """
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return X


def find_rv(X):
    """Execute in each party locally.
    """
    R = np.dot(X.T, X)
    v = np.sum(X, axis=0)
    N = len(X)
    
    return R, v, N


def join_cov(R_list, v_list, N_list):
    """Calculate the joint covariance matrix from multi-party preprocess.
    """
    s = np.sum(R_list, axis=0)
    print("shape: ", s.shape)
    v = np.sum(v_list, axis=0).reshape((len(s), 1))
    N = np.sum(N_list)
    v /= np.sqrt(N)
    print("changed")

    cov = (s - np.dot(v,v.T))/(N-1)
    
    return cov


def cov_generation(X, flag_large=False):
    """Generate the coveriance matrix.
    """
    mean_x = np.mean(X, axis=0)
    if flag_large:
        cov = mult_block(X-mean_x, k=3)/(X.shape[0]-1)
    else:
        cov = np.dot((X-mean_x).T, (X-mean_x)) / (X.shape[0]-1)
    return cov


def projection_construction(X, cov_flag=True):
    """Construct the full projection matrix with the covariance matrix.

    cov_X: covariance matrix
    """
    if cov_flag:
        eig, eigv, _, _ = jacobi_loop(X)
    else:
        X = cov_generation(X)
        eig, eigv, _, _ = jacobi_loop(X)
        
    eig_arg = np.argsort(np.abs(eig))[::-1]
    p_matrix = eigv[:, eig_arg]

    return p_matrix


def delete_replicate(A, B):
    """Delete the replicated samples from the larger dataset.
    """
    
    n, m, d = len(A), len(B), A.shape[1]
    mask = np.ones(max(n, m))
    
    for i in range(d):
        comp_res = (A.T[i].repeat(len(B)).reshape(len(A), len(B)) == B.T[i])
        mask *= comp_res.sum(axis=0)
    
    mask = (mask == 0).reshape((len(mask), 1))
    print(mask.shape)
    
    if(n>m):
        A *= mask
    else:
        B *= mask
    
    return A, B

# def classiification_analysis(y_pred, y_true):
#     """Generate the report about the classification model
#     """
#     precision_score = 

if __name__ == '__main__':

    # brief test
    dim = (30, 5)
    k = 3
    len_each = dim[0]//k

    data = np.random.random(dim)
    data_list = [data[i*len_each:(i+1)*len_each] for i in range(k)]

    cov = cov_generation(data) # calculate the cov for fused data
    R_list, v_list = [], []
    N_list = []

    for X in data_list: # calculate locally and join in the platform
        R, v, N = find_rv(X)
        R_list.append(R)
        v_list.append(v)
        N_list.append(N)

    cov2 = join_cov(R_list, v_list, N_list)
    np.testing.assert_almost_equal(cov2, cov)
    print("CONGRALUTIONS")