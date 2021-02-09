"""
QR with Householder reflection functions
Author: fanxy20@mails.tsinghua.edu.cn
"""

import time
import numpy as np

def house_holder_reduction(A):
    """Reduce the origional matrix to upper hessenberg matrix.
    """
    n = len(A)
    Q = np.identity(n)
    R = A.copy()
    
    till=n-2

    for j in range(0, till):
        
        # update R each round.
        x = R[j+1:n, j].copy()
        u = R[j+1:n, j].copy().reshape((n-j-1, 1))
        u_norm = np.linalg.norm(u) # sqrt*1

        sign_u = (u[0]>0)*2-1
        u[0] = u[0] - sign_u*u_norm # comparision*1
        v = u/np.linalg.norm(u)#sqrt*1

        P = np.identity(n-j-1) - 2*np.dot(v, v.T) # dot*4; Q in each round
        tmp = np.identity(n)
        tmp[j+1:n, j+1:n] = P
        
        Q = np.dot(tmp, Q)
        
        #R[j+1:n, j:n] = np.dot(P, R[j+1:n, j:n])
        #R[j:n, j+1:n] = np.dot(R[j:n, j+1:n], P.T)
        R = np.dot(np.dot(tmp, R), tmp.T)
        #print("r: ", R.round(3))
    res = np.dot(np.dot(Q, A), Q.T)  
    
    return res, Q

def qr(A):
    """
    for tridiangle matrix - based on house holder reflection.
    """
    n = len(A)
    Q = np.identity(n)
    #R = A.copy()
    R = np.array(A)

    for i in range(0, n-1):
        x = R[i:, i].copy().reshape((n-i, 1))
    
        x_norm = np.linalg.norm(x) # norm*1
        sign_x = (x[0]>0)*2-1
        x[0] = x[0] - sign_x*x_norm
    
        v = x/np.linalg.norm(x) # norm*2
        P = np.identity(n-i) - 2*np.dot(v, v.T)

        tmp = np.identity(n)
        tmp[i:n, i:n] = P

        R = np.dot(tmp, R)
        Q = np.dot(Q, tmp)
    
    return Q, R

def eigen_decomposition_qr_shifts(A, tol=1e-6):
    """Calculate the eigen value using LAPACK style.
    """
    iters = 0
    n = len(A)
    vals = np.zeros(n)
    T, Q = house_holder_reduction(A)
    vecs = Q.T
    I_VEC = np.identity(n)
    for m in range(n-1, 0, -1):

        I = np.identity(len(T))
        while(True):
            iters += 1

            mu = T[-1, -1]
            Q, R = qr(T - mu*I)
            T = np.dot(R, Q) + mu*I
            Q_v = I_VEC.copy()
            Q_v[:len(Q), :len(Q)] = Q
            vecs = np.dot(vecs, Q_v)

            off_diag = T[-1, -2]**2
            if off_diag/n < tol:
                vals[m] = T[-1, -1]
                T = T[:-1, :-1]
                break
                
    vals[0] = T[0, 0]
    print("ROUNDS: ", iters*(n-1))
    return vals, vecs, iters


if __name__ == '__main__':
    # test for correctness
    N_list = [30, 50, 70, 90, 100, 120, 150, 200]
    #N_list = [500]
    conv = np.zeros((len(N_list), 50))
    for i in range(50):
        for j in range(len(N_list)):
            np.random.seed(i)
            
            #tmp = np.random.normal(0, 10, (N, N))
            #tmp = np.random.normal(5, 5, (N_list[j], N_list[j]))
            tmp = np.random.uniform(0, 5, (N_list[j], N_list[j]))
            #tmp = np.ones((N, N))
            A = tmp + tmp.T
            # print("Estimated rounds: ", e_iters)
            evals, evecs, iter_num = eigen_decomposition_qr_shifts(A)
            #print(">>>>>>>>>>>>", sums_o5ff)
            conv[j][i] = iter_num
            print("Real rounds: ", iter_num)
            # for c in range(A.shape[0]):
            #     np.testing.assert_almost_equal(np.dot(A, evecs[:, c]), evecs[:, c] * evals[c], decimal=3)
            #print("time: ", time_all)
    print("mean: ", np.mean(conv, axis=1))

