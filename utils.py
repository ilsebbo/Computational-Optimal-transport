# Implementation of our algorithm for rounding a matrix onto the U_{r,c}
# transport polytope. See Section 2 of the paper for details.

import numpy as np
import scipy.linalg as spl

def K_mult_vec(V, L, x):
    """
    Multiply the kernel matrix K_tilde by a vector x with complexity O(nr).
    Actually, the true complexity is O(nr + r^2) but we can assume that r << n.

    Parameters:
        V (np.ndarray): Change of basis matrix (n x r).
        L (np.ndarray): Cholesky factor of the kernel matrix (r x r).
        x (np.ndarray): Vector (n-dimensional).
        
    Returns:
        y (np.ndarray): Result of the multiplication (n-dimensional).
    """
    Vtx = V.T @ x # Complexity: O(n r)
    y = np.linalg.solve(L, Vtx) # Complexity: O(r^2)
    y_prime = np.linalg.solve(L.T, y) # Complexity: O(r^2)

    return V @ y_prime # Complexity: O(n r)


def K_round(V, L, D1, D2, p, q, bool_round = True):
    """
    Round the kernel matrix D1 K_tilde D2 onto the U_{r,c} transport polytope.

    Parameters:
        V (np.ndarray): Change of basis matrix (n x r).
        L (np.ndarray): Cholesky factor of the kernel matrix (r x r).
        D1 (np.ndarray): Positive diagonal matrix (rows).
        D2 (np.ndarray): Positive diagonal matrix (columns).
        p (np.ndarray): Target row sums (n-dimensional vector).
        q (np.ndarray): Target column sums (n-dimensional vector).
        
    Returns:
        K_hat (np.ndarray): Rounded kernel matrix (n x n).
    """
    n = len(p)

    # sum of the rows of D1 K_tilde D2
    r0 = D1 * K_mult_vec(V, L, D2) # Complexity: O(n r)

    # minimum component-wise between p/r0 and 1
    x = np.minimum(p/r0, 1) # Complexity: O(n)

    # sum of the columns of K' = diag(x) D1 K_tilde D2
    D1x = D1 * x # Complexity: O(n)
    c1 = D2 * K_mult_vec(V, L, D1x) # Complexity: O(n r)

    # minimum component-wise between q/c1 and 1
    y = np.minimum(q/c1, 1) # Complexity: O(n)

    # sum of the rows of K'' = K' diag(y) = diag(x) D1 K_tilde D2 diag(y)
    D2y = D2 * y # Complexity: O(n)
    r2 = D1 * K_mult_vec(V, L, D2y) # Complexity: O(n r)

    # sum of the columns of K''
    c2 = y * c1 # Complexity: O(n)

    # error vectors
    err_r = p - r2 # Complexity: O(n)
    err_c = q - c2 # Complexity: O(n)

    # Until this point the complexity is O(n r)

    # return the rounded kernel matrix
    invL = np.linalg.inv(L) # Complexity: O(r^3)
    K_tilde = V @ invL.T @ invL @ V.T # Complexity: O(n r^2) 
    P_tilde = np.diag(D1) @ K_tilde @ np.diag(D2) # Complexity: O(n r)   

    if not bool_round:
        return P_tilde
    
    P_hat = P_tilde + np.outer(err_r, err_c) / np.sum(np.abs(err_r)) # Complexity: O(n^2)

    return P_hat
    

def sinkhorn(V, L, p, q, delta):
    """
    Sinkhorn algorithm to compute positive diagonal matrices D1, D2 
    and the cost W_hat.
    
    Parameters:
        V (np.ndarray): Change of basis matrix (n x r).
        L (np.ndarray): Cholesky factor of the kernel matrix (r x r).
        p (np.ndarray): Target row sums (n-dimensional vector).
        q (np.ndarray): Target column sums (n-dimensional vector).
        delta (float): Convergence tolerance parameter.
        
    Returns:
        D1 (np.ndarray): Positive diagonal matrix (rows).
        D2 (np.ndarray): Positive diagonal matrix (columns).
        W_hat (float): Cost.
    """
    n = len(p)
    tau = delta / 8  
    D1 = np.ones(n)  # Initialize D1 as the diagonal of the identity matrix
    D2 = np.ones(n)  # Initialize D2 as the diagonal of the identity matrix
    k = 0
    min_dev = 5000
    max_iter = 100000

    # Step 2: Round p and q
    p_prime = (1 - tau) * p + tau / n # Complexity O(n)
    q_prime = (1 - tau) * q + tau / n # Complexity O(n)

    # invL = np.linalg.inv(L)
    # K_tilde = V @ invL.T @ invL @ V.T


    while True:  # Step 3
        # Compute row and column deviations

        # cond_p = np.linalg.norm(D1 * (K_tilde @ D2) - p_prime, ord=1)
        K_d2 = K_mult_vec(V, L, D2) # Complexity: O(n r)
        cond_p = np.linalg.norm(D1 * K_d2 - p_prime, ord=1) # Complexity: O(n)


        # cond_q = np.linalg.norm(D2 * (K_tilde.T @ D1) - q_prime, ord=1)
        K_d1 = K_mult_vec(V, L, D1) # Complexity: O(n r)
        cond_q = np.linalg.norm(D2 * K_d1 - q_prime, ord=1) # Complexity: O(n)
        deviation = cond_p + cond_q # Complexity: O(n)

        # update the min
        if deviation < min_dev:
            min_dev = deviation # Complexity: O(1)

        # Step 4: Check convergence      
        if deviation <= delta / 2:
            print(f"Sinkhorn converged in {k} iterations with deviation {deviation} and tolerance {delta/2} ")
            break
        elif k > max_iter and np.abs(deviation - min_dev) < delta:
            print(f"Sinkhorn reached max_iter {max_iter} with deviation {deviation} and tolerance {delta/2} with min_dev {min_dev} so far")
            break

        if k % 5000 == 0:
            print(f"Iteration {k} with deviation {deviation} and tolerance {delta/2} with min_dev {min_dev} so far")
        
        # check if K_d1 or K_d2 have any zero entries
        if np.any(K_d1 == 0) or np.any(K_d2 == 0):
            print("Zero entries in K_d1 or K_d2")
            break

        k += 1

        if k % 2 == 1:  # Step 5: Renormalize rows
                       
            D1 = p_prime / K_d2 # Complexity: O(n)

        else:  # Step 7: Renormalize columns
            
            D2 = q_prime / K_d1 # Complexity: O(n)

    # Step 11: Compute cost
    W_hat = 0
    K_d2 = K_mult_vec(V, L, D2) # Complexity: O(n r)
    K_d1 = K_mult_vec(V, L, D1) # Complexity: O(n r)

    W_hat += np.sum(np.log(D1) * (D1 * K_d2)) # Complexity: O(n)
    W_hat += np.sum(np.log(D2) * (D2 * K_d1)) # Complexity: O(n)


    return D1, D2, W_hat 


def sinkhorn_iter(V, L, p, q, N_iter, eta, C):
    """
    Sinkhorn algorithm to compute positive diagonal matrices D1, D2 
    and the cost W_hat.
    
    Parameters:
        V (np.ndarray): Change of basis matrix (n x r).
        L (np.ndarray): Cholesky factor of the kernel matrix (r x r).
        p (np.ndarray): Target row sums (n-dimensional vector).
        q (np.ndarray): Target column sums (n-dimensional vector).
        N_iter (int): Number of iterations.
        eta (float): Kernel parameter. (only for the cost computation)
        C (np.ndarray): Cost matrix. (only for the cost computation)
        
    Returns:
        D1 (np.ndarray): Positive diagonal matrix (rows).
        D2 (np.ndarray): Positive diagonal matrix (columns).
        W_hat (float): Cost.
    """
    n = len(p)
    # tau = delta / 8  
    D1 = np.ones(n)  # Initialize D1 as the diagonal of the identity matrix
    D2 = np.ones(n)  # Initialize D2 as the diagonal of the identity matrix
    k = 0
    min_dev = 5000
    max_iter = 100000
    inter = int(N_iter / 5)

    Sink_cost = []
    Sink_cost.append(compute_Sink_cost(V, L, D1, D2, p, q, eta, C))
    
    # Step 2: Round p and q
    tau = 0
    p_prime = (1 - tau) * p + tau / n
    q_prime = (1 - tau) * q + tau / n

    # invL = np.linalg.inv(L)
    # K_tilde = V @ invL.T @ invL @ V.T


    for k in range(N_iter):  # Step 3 
        # Compute row and column deviations

        # cond_p = np.linalg.norm(D1 * (K_tilde @ D2) - p_prime, ord=1)
        K_d2 = K_mult_vec(V, L, D2)
        cond_p = np.linalg.norm(D1 * K_d2 - p_prime, ord=1)


        # cond_q = np.linalg.norm(D2 * (K_tilde.T @ D1) - q_prime, ord=1)
        K_d1 = K_mult_vec(V, L, D1)
        cond_q = np.linalg.norm(D2 * K_d1 - q_prime, ord=1)
        deviation = cond_p + cond_q

        # update the min
        if deviation < min_dev:
            min_dev = deviation

        if k % inter == 0 and k>0:
            cost = compute_Sink_cost(V, L, D1, D2, p, q, eta, C)
            Sink_cost.append(cost)
            print(f"Iteration {k} with deviation {deviation} with cost {cost}")

        
        # check if K_d1 or K_d2 have any zero entries
        if np.any(K_d1 <= 0) or np.any(K_d2 <= 0):
            print("Zero entries in K_d1 or K_d2 with minimum value: ", np.min(K_d1), np.min(K_d2))
            # print(f"skipping iteration {k}")
            # break


        if k % 2 == 1:  # Step 5: Renormalize rows
                       
            D1 = p_prime / K_d2

        else:  # Step 7: Renormalize columns
            
            D2 = q_prime / K_d1


    # Sink_cost.append(compute_Sink_cost(V, L, D1, D2, p, q, eta, C))

    # Step 11: Compute cost
    W_hat = 0
    K_d2 = K_mult_vec(V, L, D2)
    K_d1 = K_mult_vec(V, L, D1)
    
    # if D1 or D2 < 1e-05 equal to 1
    D1_p = D1.copy()
    D1_p[ D1 < 1e-05] = 1
    D2_p = D2.copy()
    D2_p[ D2 < 1e-05] = 1

    W_hat += np.sum(np.log(D1_p) * (D1 * K_d2))
    W_hat += np.sum(np.log(D2_p) * (D2 * K_d1))

    print(f"Final deviation {deviation} and estimated cost {W_hat}")


    return D1, D2, W_hat, Sink_cost



def Classic_Sinkhorn(eta, C, p, q, N_iter = 5000):

    K = np.exp(-eta*C)
    Err_q = []
    Err_p = []
    v = np.ones(len(q))
    u = np.ones(len(p))

    inter = int(N_iter / 5)

    Sink_cost = []

    # Only to compute the cost
    P_hat = np.diag(u) @ K @ np.diag(v)
    P_hat_zero = np.where(P_hat < 1e-10, 1, P_hat)
    Sink_cost.append(np.sum(np.multiply(P_hat, C)) + 1/eta * np.sum(np.multiply(P_hat_zero, np.log(P_hat_zero))))


    for k in range(N_iter):
        
        # evaluate the cost
        if k % inter == 0 and k>0:
            P_hat = np.diag(u) @ K @ np.diag(v)
            P_hat_zero = np.where(P_hat < 1e-10, 1, P_hat)
            cost = np.sum(np.multiply(P_hat, C)) + 1/eta * np.sum(np.multiply(P_hat_zero, np.log(P_hat_zero)))
            Sink_cost.append(cost)

        # sinkhorn step 1
        u = p / (np.dot(K,v)) # Complexity: O(n^2)

        # error computation
        r = v*np.dot(np.transpose(K),u) # Complexity: O(n^2)
        Err_q = Err_q + [np.linalg.norm(r - q, 1)]

        # sinkhorn step 2
        v = q /(np.dot(np.transpose(K),u)) # Complexity: O(n^2)

        # error computation
        s = u*np.dot(K,v)
        Err_p = Err_p + [np.linalg.norm(s - p,1)]

        if k % inter == 0 and k>0:
            deviation = Err_p[-1] + Err_q[-1]
            print(f"Iteration {k} with deviation {deviation} with cost {cost}")

    P_hat = np.diag(u) @ K @ np.diag(v)
    P_hat_zero = np.where(P_hat < 1e-10, 1, P_hat)
    cost = np.sum(np.multiply(P_hat, C)) + 1/eta * np.sum(np.multiply(P_hat_zero, np.log(P_hat_zero)))
    Sink_cost.append(cost)
    
    return u, v, Sink_cost



###################### Nystrom ############################

def gauss(X: np.ndarray, Y: np.ndarray=None, eta=0.01):
    """
    Gaussian kernel function.

    Parameters:
        X (np.ndarray): Input data matrix (n x d).
        Y (np.ndarray): Input data matrix (m x d).
        gamma (float): Kernel parameter.

    Returns:
        Ksub (np.ndarray): Kernel matrix (n x m).
    """

    if Y is None:
        Ksub = np.ones((X.shape[0], 1))
    else:
        nsq_rows = np.sum(X ** 2, axis=1, keepdims=True)
        nsq_cols = np.sum(Y ** 2, axis=1, keepdims=True)
        Ksub = nsq_rows - np.matmul(X, Y.T * 2)
        Ksub = nsq_cols.T + Ksub
        Ksub = np.exp(-eta * Ksub)

    return Ksub


def recursiveNystrom(X, n_components: int, kernel_func = gauss, eta = 0.01, accelerated_flag=False, random_state=None, lmbda_0=0, return_leverage_score=False, **kwargs):
    '''
    Recursive Nystrom sampling algorithm for kernel matrix approximation with leverage score sampling.
    Code provided by the authors of Recursive Sampling for the Nystrom Method from https://github.com/axelv/recursive-nystrom/blob/master/recursive_nystrom.py.

    Parameters:
        X (np.ndarray): Input data matrix (n x d).
        n_components (int): Number of columns to sample.
        kernel_func: Kernel function.
        eta (float): gamma parameter of the kernel function.
        accelerated_flag: Accelerated version of the algorithm.
        random_state: Random seed.
        lmbda_0: Regularization parameter.
        return_leverage_score: Return leverage scores.
        kwargs: Additional arguments.

    Returns:
        indices (np.ndarray): Subset of indices (r x 1).
        leverage_score (np.ndarray): Leverage scores.
    '''

    rng = np.random.RandomState(random_state)

    n_oversample = np.log(n_components)
    k = np.ceil(n_components / (4 * n_oversample)).astype(int)
    n_levels = np.ceil(np.log(X.shape[0] / n_components) / np.log(2)).astype(int)
    perm = rng.permutation(X.shape[0])

    # set up sizes for recursive levels
    size_list = [X.shape[0]]
    for l in range(1, n_levels+1):
        size_list += [np.ceil(size_list[l - 1] / 2).astype(int)]

    # indices of points selected at previous level of recursion
    # at the base level it's just a uniform sample of ~ n_component points
    sample = np.arange(size_list[-1])
    indices = perm[sample]
    weights = np.ones((indices.shape[0],))

    # we need the diagonal of the whole kernel matrix, so compute upfront
    k_diag = kernel_func(X)

    # Main recursion, unrolled for efficiency
    for l in reversed(range(n_levels)):
        # indices of current uniform sample
        current_indices = perm[:size_list[l]]
        # build sampled kernel

        # all rows and sampled columns
        KS = kernel_func(X[current_indices,:], X[indices,:], eta)
        SKS = KS[sample, :] # sampled rows and sampled columns

        # optimal lambda for taking O(k log(k)) samples
        if k >= SKS.shape[0]:
            # for the rare chance we take less than k samples in a round
            lmbda = 10e-6
            # don't set to exactly 0 to avoid stability issues
        else:
            # eigenvalues equal roughly the number of points per cluster, maybe this should scale with n?
            # can be interpret as the zoom level
            lmbda = (np.sum(np.diag(SKS) * (weights ** 2)) - np.sum(spl.eigvalsh(SKS * weights[:,None] * weights[None,:], eigvals=(SKS.shape[0]-k, SKS.shape[0]-1))))/k
        lmbda = np.maximum(lmbda_0*SKS.shape[0], lmbda)
        if lmbda == lmbda_0*SKS.shape[0]:
            print("Set lambda to %d." % lmbda)
        #lmbda = np.minimum(lmbda, 5)
            # lmbda = spl.eigvalsh(SKS * weights * weights.T, eigvals=(0, SKS.shape[0]-k-1)).sum()/k
            # calculate the n-k smallest eigenvalues

        # compute and sample by lambda ridge leverage scores
        R = np.linalg.inv(SKS + np.diag(lmbda * weights ** (-2)))
        R = np.matmul(KS, R)
        #R = np.linalg.lstsq((SKS + np.diag(lmbda * weights ** (-2))).T,KS.T)[0].T
        if l != 0:
            # max(0, . ) helps avoid numerical issues, unnecessary in theory
            leverage_score = np.minimum(1.0, n_oversample * (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
            # on intermediate levels, we independently sample each column
            # by its leverage score. the sample size is n_components in expectation
            sample = np.where(rng.uniform(size=size_list[l]) < leverage_score)[0]
            # with very low probability, we could accidentally sample no
            # columns. In this case, just take a fixed size uniform sample
            if sample.size == 0:
                leverage_score[:] = n_components / size_list[l]
                sample = rng.choice(size_list[l], size=n_components, replace=False)
            weights = np.sqrt(1. / leverage_score[sample])

        else:
            leverage_score = np.minimum(1.0, (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
            p = leverage_score/leverage_score.sum()

            sample = rng.choice(X.shape[0], size=n_components, replace=False, p=p)
        indices = perm[sample]

    if return_leverage_score:
        return indices, leverage_score[np.argsort(perm)]
    else:
        return indices
    

def Nystrom(X, indices, kernel_func = gauss, eta = 0.01):

    """
    Nystrom takes the input data matrix X and the subset of indices to compute the Nystrom approximation.

    Parameters:
        X (np.ndarray): Input data matrix (n x d).
        indices (np.ndarray): Subset of indices (r x 1).
        kernel_func: Kernel function.
        eta (float): gamma parameter of the kernel function.

    Returns:
        V (np.ndarray): change of basis matrix (n x r).
        L (np.ndarray): Cholesky factor of the kernel matrix (r x r).        
    """

    submatrix = X[indices, :]
    V = kernel_func(X, submatrix, eta)
    A = kernel_func(submatrix, submatrix, eta)
    L = np.linalg.cholesky(A) # Complexity: O(r^3)

    return V, L

compute_K_tilde = lambda V, L: V @ np.linalg.inv(L).T @ np.linalg.inv(L) @ V.T

def compute_Sink_cost(V, L, D1, D2, p, q, eta, C):

    P_hat = K_round(V, L, D1, D2, p, q)
    # to impose 0*log(0) = 0
    # print(f'P_hat has {np.sum(P_hat < 0)/p.shape[0]**2 *100}% negative values with minimum value: ', np.min(P_hat))
    # print(f'P_hat_zero has {np.sum(P_hat_zero == 0)/p.shape[0]**2 *100}% zero values with minimum value: ', np.min(P_hat_zero))
    P_hat_zero = np.where(P_hat < 1e-10, 1, P_hat)
    return np.sum(np.multiply(P_hat, C)) + 1/eta * np.sum(np.multiply(P_hat_zero, np.log(P_hat_zero)))
    
def Nystrom_LS(X, n_components: int, kernel_func = gauss, eta = 0.01, random_state=None, **kwargs):
    """
    Adaptive Nystrom sampling algorithm for kernel matrix approximation.

    Parameters:
        X (np.ndarray): Input data matrix (n x d).
        n_components (int): Number of columns to sample.
        kernel_func: Kernel function.
        eta (float): gamma parameter of the kernel function.
        random_state: Random seed.
        kwargs: Additional arguments.

    Returns:
        V (np.ndarray): Change of basis matrix (n x r).
        L (np.ndarray): Cholesky factor of the kernel matrix (r x r).
        r (int): Number of columns sampled.
    """

    indices = recursiveNystrom(X, n_components, kernel_func, eta, random_state=random_state, **kwargs)
    V, L = Nystrom(X, indices, kernel_func, eta)

    return V, L, n_components

def AdaptiveNystrom(X, eta, tau):
    '''
    Adaptive Nystrom sampling algorithm for kernel matrix approximation with 
    leverage score sampling and doubling trick.

    Parameters:
        X (np.ndarray): Input data matrix (n x d).
        eta (float): Kernel parameter.
        tau (float): Convergence tolerance parameter.

    Returns:
        V (np.ndarray): Change of basis matrix (n x r).
        L (np.ndarray): Cholesky factor of the kernel matrix (r x r).
        rnk (int): rank of the approximation.
    '''
    r = 1
    V = np.zeros((X.shape[0], 1))
    L = np.zeros((1, 1))
    while True:

        r = 2 * r # Double the number of columns sampled

        # Compute Nystrom approximation
        V, L, r= Nystrom_LS(X, r, gauss, eta)

        # Compute the l-infinity component-wise error in an efficient way 

        v_i = np.linalg.solve(L, V.T) # Solve the linear system L v_i = V_i
        norm_i = np.linalg.norm(v_i, axis=0) # Compute the norm of each column of v_i
        error = 1 - np.min(norm_i) # Compute the error

        if error < tau:
            break
    
    rnk = np.linalg.matrix_rank(L)
    return V, L, rnk





# n = 200 # number of points
# d = 1 # dimensionality
# eta = 5


# X = 0.7*np.random.rand(n, d) # generate random data

# K = gauss(X, X, eta) # compute the kernel matrix

# # check if K has negative values
# if np.any(K < 0):
#     print(f"K has {np.sum(K < 0)/n**2 *100}% negative values with minimum value: ", np.min(K))
#     # check if K is positive semi-definite
#     # eigs = np.linalg.eigvals(K)
#     # if np.any(eigs < 0):
#     #     print("K is not positive semi-definite with minimum eigenvalue: ", np.min(eigs))

# V, L, r = Nystrom_LS(X, 10, gauss, eta) # compute the Nystrom approximation

# K_tilde = compute_K_tilde(V, L) # compute the kernel matrix approximation

# # check if K_tilde has negative values
# if np.any(K_tilde < 0):
#     print(f"K_tilde has {np.sum(K_tilde < 0)/n**2 *100}% negative values with minimum value: ", np.min(K_tilde))
#     # check if K_tilde is positive semi-definite
#     # eigs = np.linalg.eigvals(K_tilde)
#     # if np.any(eigs < 0):
#     #     print("K_tilde is not positive semi-definite with minimum eigenvalue: ", np.min(eigs))
# else:
#     print("K_tilde has no negative values")


