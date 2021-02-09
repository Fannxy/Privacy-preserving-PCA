"""
Application experiments for eigen decomposition
"""
import numpy as np
import pandas as pd
import time
import scipy.io as scio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
import matplotlib.pyplot as plt

from jacobi import *
from qr import *
from plain_preprocess import *

def abnormal_detection_pca(X, K, d_ratio=90, p_matrix=None):
    """abnormally detection based on PCA
    """
    if p_matrix is None:
        p_matrix, X_reduce = dimension_reduction(X, K)
    else:
        p_matrix = p_matrix[:, :K]
        X_reduce = np.dot(X, p_matrix)
    X_recon = np.dot(X_reduce, p_matrix.T)
    recon_error = np.sum(np.sqrt((X - X_recon)**2), axis=1)
    d = np.percentile(recon_error, d_ratio)
    y_pred = recon_error > d

    return y_pred

def dimension_reduction(X, k=10):
    """Reduction dimension to ananlyze the data.
    Reduce the dimension of X to k
    """
    cov = cov_generation(X)
    eig, eigv, _, _ = jacobi_loop(cov)
    sort_args = np.argsort(np.abs(eig))[::-1]
    projection_matrix = eigv[sort_args][:, :k]
    reduce_x = np.dot(X, projection_matrix)
    
    return projection_matrix, reduce_x


def dimension_reduction_np(X, k=10):
    """Implement the same version using Numpy
    """
    cov = cov_generation(X)
    eig, eigv = np.linalg.eig(cov)
    sort_args = np.argsort(np.abs(eig))[::-1]
    projection_matrix = np.real(eigv[sort_args][:, :k])
    reduce_x = np.dot(X, projection_matrix)

    return projection_matrix, reduce_x


def detection_analysis(y_pred, y_true):
    """print the correspondnig scores
    """
    print("Precision: ", sm.precision_score(y_pred, y_true))
    print("Recall: ", sm.recall_score(y_pred, y_true))
    print("Accuracy: ", sm.accuracy_score(y_pred, y_true))
    print("\n")

if __name__ == '__main__':

    # Outlier detection
    print("========= Experiments 1 Musk detection ========")
    data_musk = scio.loadmat('./Data/Musk/musk.mat')
    data_musk = np.concatenate([data_musk['X'], data_musk['y']], axis=1)
    np.random.seed(8)
    print(data_musk.shape)
    np.random.shuffle(data_musk)
    K = 105 # hyper-parameter
    X = data_musk[:, :-1] # 3602*166
    y = data_musk[:, -1] # 1-outlier, 0-normal; proportion: 0.03168; 97 outliers totally

    print(">>>> benchmark - full dataset")
    print("K = ", K)
    y_pred = abnormal_detection_pca(X, K, d_ratio=95)
    detection_analysis(y_pred, y)

    L = 2
    samples_m = len(X) // 2
    for i in range(L):
        print("Party - ", i)
        X_local = X[i*samples_m:(i+1)*samples_m]
        y_local = y[i*samples_m:(i+1)*samples_m]
        y_pred = abnormal_detection_pca(X_local, K, d_ratio=95)
        print("sums: ", np.sum(y_pred))
        detection_analysis(y_pred, y_local)




