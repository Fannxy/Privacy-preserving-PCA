"""
Jaocbi methods
Author:fanxy20@mails.tsinghua.edu.cn
"""
import time
import numpy as np


def estimate_rounds(A):
    """Using to estimate the convergence rounds for A
    """
    n = len(A)
    a_offmean = (np.sum(A**2) - np.sum(np.diagonal(A)**2))/(n*(n-1))
    print(a_offmean)

    e_iters = (n-1)*np.log((n-1)*a_offmean)

    return e_iters

# Decomposition base on givens rotation
def rotate(a, i):
    """Rotate based on parallel givens.
    """
    n = len(a)
    def _jacobi_set_selection(a, i):
        n = len(a)
        m = (n+1) // 2
        l_list = []

        if i <= m-1:
            k_list = [m-i+j for j in range(0, n-m)]
            for k in k_list:
                # without -1
                if (k <= 2*(m-i)-1):
                    l_list.append(2*(m-i)-k-1)
                elif (k<= 2*m-i-2):
                    l_list.append(2*(2*m-i-1)-k)
                else:
                    l_list.append(n-1)
        else:
            k_list = [4*m-n-i+j-1 for j in range(0, n-m)]
            for k in k_list:
                # without -1
                if (k<2*m-i):
                    l_list.append(n-1)
                elif (k<= 2*(2*m-i-1)):
                    l_list.append(2*(2*m-i-1)-k)
                else:
                    l_list.append(2*(3*m-i)-3-k)
                
        k_list = np.array(k_list)
        l_list = np.array(l_list)
        return k_list, l_list
    
    k_list, l_list = _jacobi_set_selection(a, i)

    # Calculate the rotate angle.
    a_diff = a[k_list, k_list] - a[l_list, l_list]
    
    # For two calculation strategy, avoiding the zero denominator
    flag_zero = a[k_list, l_list]**2 < a_diff**2 * 1e-10
    phi = a_diff / (2 * a[k_list, l_list])
    sign_phi = (phi > 0) * 2 - 1
    tan = flag_zero * (a[k_list, l_list] / (a_diff + 1 - flag_zero)) + (
        1 - flag_zero) * (sign_phi / (phi * sign_phi + np.sqrt(phi**2 + 1.0)))
    cos = 1.0 / np.sqrt(tan**2 + 1.0)
    sin = tan * cos

    # Calculate the batch rotation matrix Q
    Q = np.eye(n) + 0
    for i in range(len(k_list)):
        Q[k_list[i], k_list[i]] = cos[i]
        Q[l_list[i], l_list[i]] = cos[i]
        Q[k_list[i], l_list[i]] = -sin[i]
        Q[l_list[i], k_list[i]] = sin[i]
    Q_T = np.transpose(Q)

    return Q, Q_T


def jacobi_loop(a, tol=1.0e-5):  # Jacobi method
    """Cylic-by-row jacobi
    """
    sums_off_list = []
    a = a.copy()
    n = a.shape[0]
    assert n > 1
    m = (n+1)//2
    
    LOOP_INTERVAL = 3 # The iteration numbers for convergence check
    MAX_ROT = 5*n**2 # Set limit on number of rotations
    eigen_vecs = np.eye(n)
    iter_num = 0
    sweep_num = 0
    for i in range(MAX_ROT):  # Jacobi rotation loop
        sweep_num += 1
        for j in range(1, 2*m):
            iter_num += 1
            Q, Q_T = rotate(a, j)
            a = np.dot(np.dot(Q_T, a), Q)
            eigen_vecs = np.dot(eigen_vecs, Q)
            eigen_vals = np.diagonal(a)
            #print(a)
            if iter_num%LOOP_INTERVAL == 0:
                # sums_all = np.sum(a*a)
                # sums_diag = np.sum(eigen_vals*eigen_vals)
                sums_off = (np.sum(a**2) - np.sum(np.diagonal(a)**2))
                sums_off_list.append(sums_off)

                if sums_off/n < (tol):
                    print("loop - iterations: ", iter_num)
                    print("sweep - iterations: ", sweep_num)
                    return eigen_vals, eigen_vecs, iter_num, sums_off_list
        

    raise Exception('Jacobi method did not converge')


def svd_jacobi(g, tol=1.0e-6):  # Jacobi method

    g = g.copy()
    n = g.shape[0]
    assert n > 1
    m = (n+1)//2
    
    LOOP_INTERVAL = 1 # The iteration numbers for convergence check
    MAX_ROT = 5*(n**2) # Set limit on number of rotations
    
    start_flag = False
    iter_num = 0
    for i in range(MAX_ROT):
        for j in range(1, 2*m):
            iter_num += 1
            a = np.dot(g.T, g)
            Q,_ = rotate(a, j)
            g = np.dot(g, Q)
                
            if start_flag:
                q = np.dot(q, Q)
            else:
                q = Q
            start_flag = True
                
            if iter_num % LOOP_INTERVAL == 0:
                res_a = np.dot(g.T, g)
                sums_all = np.sum(np.ravel(res_a**2))
                sums_diag = np.sum([res_a[j, j]**2 for j in range(len(g))])
                
                if sums_all - sums_diag < tol:
                    print("SVD - loop - iters: ", iter_num)
                    sig = np.sqrt(np.sum(g**2, axis=0))
                    U = g / sig
                    V = q
                    return np.diag(sig), U, V
    

    raise Exception('Jacobi method did not converge')


if __name__ == "__main__":

    # # test correctness
    # N = 34

    # test convergence for jacobi-eigendecomposition
    N_list = [30, 50, 70, 90, 100, 120, 150, 200]
    #N_list = [10, 15, 20]
    conv = np.zeros((len(N_list), 50))
    for i in range(50):
        for j in range(len(N_list)):
            np.random.seed(i)
            
            #tmp = np.random.normal(0, 10, (N, N))
            #tmp = np.random.uniform(0, 100, (N_list[j], N_list[j]))
            tmp = np.random.normal(50, 50, (N_list[j], N_list[j]))
            #tmp = np.ones((N, N))
            A = tmp + tmp.T
            e_iters = estimate_rounds(A)
            # print("Estimated rounds: ", e_iters)
            evals, evecs, iter_num, sums_off = jacobi_loop(A)
            #print(">>>>>>>>>>>>", sums_off)
            conv[j][i] = iter_num
            print("Real rounds: ", iter_num)
            #for c in range(A.shape[0]):
            #     np.testing.assert_almost_equal(np.dot(A, evecs[:, c]), evecs[:, c] * evals[c], decimal=3)
    print("mean: ", np.mean(conv, axis=1))


    # # test correctness for SVD
    # N = 10
    # np.random.seed(2)
    # G = np.random.random((N, N))
    # print(G)
    # sig, U, V = svd_jacobi(G)
    # print("sig = ", sig)
    # np.testing.assert_almost_equal(G, np.dot(np.dot(U, sig), V.T), decimal=6)