import cupy as cp
import scipy.linalg as spl

def K_mult_vec(V, L, x):
    """
    Multiply the kernel matrix K_tilde by a vector x with complexity O(nr).
    Actually, the true complexity is O(nr + r^2) but we can assume that r << n.

    Parameters:
        V (cp.ndarray): Change of basis matrix (n x r).
        L (cp.ndarray): Cholesky factor of the kernel matrix (r x r).
        x (cp.ndarray): Vector (n-dimensional).
        
    Returns:
        y (cp.ndarray): Result of the multiplication (n-dimensional).
    """
    Vtx = V.T @ x # Complexity: O(n r)
    y = cp.linalg.solve(L, Vtx) # Complexity: O(r^2)
    y_prime = cp.linalg.solve(L.T, y) # Complexity: O(r^2)

    return V @ y_prime # Complexity: O(n r)

def K_round(V, L, D1, D2, p, q):
    """
    Round the kernel matrix D1 K_tilde D2 onto the U_{r,c} transport polytope.

    Parameters:
        V (cp.ndarray): Change of basis matrix (n x r).
        L (cp.ndarray): Cholesky factor of the kernel matrix (r x r).
        D1 (cp.ndarray): Positive diagonal matrix (rows).
        D2 (cp.ndarray): Positive diagonal matrix (columns).
        p (cp.ndarray): Target row sums (n-dimensional vector).
        q (cp.ndarray): Target column sums (n-dimensional vector).
        
    Returns:
        K_hat (cp.ndarray): Rounded kernel matrix (n x n).
    """
    n = len(p)

    # sum of the rows of D1 K_tilde D2
    r0 = D1 * K_mult_vec(V, L, D2) # Complexity: O(n r)

    # minimum component-wise between p/r0 and 1
    x = cp.minimum(p/r0, 1) # Complexity: O(n)

    # sum of the columns of K' = diag(x) D1 K_tilde D2
    D1x = D1 * x # Complexity: O(n)
    c1 = D2 * K_mult_vec(V, L, D1x) # Complexity: O(n r)

    # minimum component-wise between q/c1 and 1
    y = cp.minimum(q/c1, 1) # Complexity: O(n)

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
    invL = cp.linalg.inv(L) # Complexity: O(r^3)
    K_tilde = V @ invL.T @ invL @ V.T # Complexity: O(n r^2)    
    K_hat = K_tilde + cp.outer(err_r, err_c) / cp.sum(cp.abs(err_r)) # Complexity: O(n^2)

    return K_hat

def sinkhorn(V, L, p, q, delta):
    """
    Sinkhorn algorithm to compute positive diagonal matrices D1, D2 
    and the cost W_hat.
    
    Parameters:
        V (cp.ndarray): Change of basis matrix (n x r).
        L (cp.ndarray): Cholesky factor of the kernel matrix (r x r).
        p (cp.ndarray): Target row sums (n-dimensional vector).
        q (cp.ndarray): Target column sums (n-dimensional vector).
        delta (float): Convergence tolerance parameter.
        
    Returns:
        D1 (cp.ndarray): Positive diagonal matrix (rows).
        D2 (cp.ndarray): Positive diagonal matrix (columns).
        W_hat (float): Cost.
    """
    n = len(p)
    tau = delta / 8  
    D1 = cp.ones(n)  # Initialize D1 as the diagonal of the identity matrix
    D2 = cp.ones(n)  # Initialize D2 as the diagonal of the identity matrix
    k = 0

    # Step 2: Round p and q
    p_prime = (1 - tau) * p + tau / n
    q_prime = (1 - tau) * q + tau / n

    while True:  # Step 3
        # Compute row and column deviations

        # cond_p = cp.linalg.norm(D1 * (K_tilde @ D2) - p_prime, ord=1)
        K_d2 = K_mult_vec(V, L, D2)
        cond_p = cp.linalg.norm(D1 * K_d2 - p_prime, ord=1)

        # cond_q = cp.linalg.norm(D2 * (K_tilde.T @ D1) - q_prime, ord=1)
        K_d1 = K_mult_vec(V, L, D1)
        cond_q = cp.linalg.norm(D2 * K_d1 - q_prime, ord=1)
        deviation = cond_p + cond_q

        # Step 4: Check convergence      
        if deviation <= delta / 2:
            print(f"Sinkhorn converged in {k} iterations with deviation {deviation} and tolerance {delta/2}")
            break
        
        k += 1

        if k % 2 == 1:  # Step 5: Renormalize rows
            D1 = p_prime / K_d2
        else:  # Step 7: Renormalize columns
            D2 = q_prime / K_d1

    # Step 11: Compute cost
    W_hat = 0
    K_d2 = K_mult_vec(V, L, D2)
    K_d1 = K_mult_vec(V, L, D1)

    W_hat += cp.sum(cp.log(D1) * (D1 * K_d2))
    W_hat += cp.sum(cp.log(D2) * (D2 * K_d1))

    return D1, D2, W_hat

def Classic_Sinkhorn(eps, C, p, q, N_iter=5000):
    """
    Classic Sinkhorn algorithm for optimal transport with entropic regularization.

    Parameters:
        eps (float): Regularization parameter.
        C (cp.ndarray): Cost matrix (n x m).
        p (cp.ndarray): Source distribution (n-dimensional).
        q (cp.ndarray): Target distribution (m-dimensional).
        N_iter (int): Number of iterations.
        
    Returns:
        u (cp.ndarray): Scaling vector for rows.
        v (cp.ndarray): Scaling vector for columns.
        Err_p (list): List of errors for p.
        Err_q (list): List of errors for q.
    """
    K = cp.exp(-C / eps)
    Err_q = []
    Err_p = []
    v = cp.ones(len(q))

    for i in range(N_iter):
        # sinkhorn step 1
        u = p / (cp.dot(K, v))

        # error computation
        r = v * cp.dot(cp.transpose(K), u)
        Err_q.append(cp.linalg.norm(r - q, ord=1))

        # sinkhorn step 2
        v = q / (cp.dot(cp.transpose(K), u))

        # error computation
        s = u * cp.dot(K, v)
        Err_p.append(cp.linalg.norm(s - p, ord=1))

    return u, v, Err_p, Err_q

###################### Nystrom ############################

def gauss(X: cp.ndarray, Y: cp.ndarray=None, gamma=0.01):
    """
    Gaussian kernel function.

    Parameters:
        X (cp.ndarray): Input data matrix (n x d).
        Y (cp.ndarray): Input data matrix (m x d).
        gamma (float): Kernel parameter.

    Returns:
        Ksub (cp.ndarray): Kernel matrix (n x m).
    """
    if Y is None:
        Ksub = cp.ones((X.shape[0], 1))
    else:
        nsq_rows = cp.sum(X ** 2, axis=1, keepdims=True)
        nsq_cols = cp.sum(Y ** 2, axis=1, keepdims=True)
        Ksub = nsq_rows - cp.matmul(X, Y.T * 2)
        Ksub = nsq_cols.T + Ksub
        Ksub = cp.exp(-gamma * Ksub)

    return Ksub

def recursiveNystrom(X, n_components: int, kernel_func=gauss, eta=0.01, accelerated_flag=False, random_state=None, lmbda_0=0, return_leverage_score=False, **kwargs):
    '''
    Recursive Nystrom sampling algorithm for kernel matrix approximation with leverage score sampling.
    Code provided by the authors of Recursive Sampling for the Nystrom Method from https://github.com/axelv/recursive-nystrom/blob/master/recursive_nystrom.py.

    Parameters:
        X (cp.ndarray): Input data matrix (n x d).
        n_components (int): Number of columns to sample.
        kernel_func: Kernel function.
        eta (float): gamma parameter of the kernel function.
        accelerated_flag: Accelerated version of the algorithm.
        random_state: Random seed.
        lmbda_0: Regularization parameter.
        return_leverage_score: Return leverage scores.
        kwargs: Additional arguments.

    Returns:
        indices (cp.ndarray): Subset of indices (r x 1).
        leverage_score (cp.ndarray): Leverage scores.
    '''
    rng = cp.random.RandomState(random_state)

    n_oversample = cp.log(n_components)
    k = cp.ceil(n_components / (4 * n_oversample)).astype(int)
    n_levels = cp.ceil(cp.log(X.shape[0] / n_components) / cp.log(2)).astype(int)
    perm = rng.permutation(X.shape[0])

    # set up sizes for recursive levels
    size_list = [X.shape[0]]
    for l in range(1, n_levels + 1):
        size_list.append(cp.ceil(size_list[l - 1] / 2).astype(int))

    # indices of points selected at previous level of recursion
    # at the base level it's just a uniform sample of ~ n_component points
    sample = cp.arange(size_list[-1])
    indices = perm[sample]
    weights = cp.ones((indices.shape[0],))

    # we need the diagonal of the whole kernel matrix, so compute upfront
    k_diag = kernel_func(X)

    # Main recursion, unrolled for efficiency
    for l in reversed(range(n_levels)):
        # indices of current uniform sample
        current_indices = perm[:size_list[l]]
        # build sampled kernel

        # all rows and sampled columns
        KS = kernel_func(X[current_indices, :], X[indices, :], eta)
        SKS = KS[sample, :]  # sampled rows and sampled columns

        # optimal lambda for taking O(k log(k)) samples
        if k >= SKS.shape[0]:
            # for the rare chance we take less than k samples in a round
            lmbda = 10e-6
            # don't set to exactly 0 to avoid stability issues
        else:
            # eigenvalues equal roughly the number of points per cluster, maybe this should scale with n?
            # can be interpret as the zoom level
            lmbda = (cp.sum(cp.diag(SKS) * (weights ** 2)) - cp.sum(spl.eigvalsh(SKS * weights[:, None] * weights[None, :], eigvals=(SKS.shape[0] - k, SKS.shape[0] - 1)))) / k
        lmbda = cp.maximum(lmbda_0 * SKS.shape[0], lmbda)
        if lmbda == lmbda_0 * SKS.shape[0]:
            print("Set lambda to %d." % lmbda)

        # compute and sample by lambda ridge leverage scores
        R = cp.linalg.inv(SKS + cp.diag(lmbda * weights ** (-2)))
        R = cp.matmul(KS, R)
        if l != 0:
            leverage_score = cp.minimum(1.0, n_oversample * (1 / lmbda) * cp.maximum(0.0, (
                    k_diag[current_indices, 0] - cp.sum(R * KS, axis=1))))
            sample = cp.where(rng.uniform(size=size_list[l]) < leverage_score)[0]
            if sample.size == 0:
                leverage_score[:] = n_components / size_list[l]
                sample = rng.choice(size_list[l], size=n_components, replace=False)
            weights = cp.sqrt(1. / leverage_score[sample])
        else:
            leverage_score = cp.minimum(1.0, (1 / lmbda) * cp.maximum(0.0, (
                    k_diag[current_indices, 0] - cp.sum(R * KS, axis=1))))
            p = leverage_score / leverage_score.sum()
            sample = rng.choice(X.shape[0], size=n_components, replace=False, p=p)
        indices = perm[sample]

    if return_leverage_score:
        return indices, leverage_score[cp.argsort(perm)]
    else:
        return indices

def Nystrom(X, indices, kernel_func=gauss, eta=0.01):
    """
    Nystrom takes the input data matrix X and the subset of indices to compute the Nystrom approximation.

    Parameters:
        X (cp.ndarray): Input data matrix (n x d).
        indices (cp.ndarray): Subset of indices (r x 1).
        kernel_func: Kernel function.
        eta (float): gamma parameter of the kernel function.

    Returns:
        V (cp.ndarray): change of basis matrix (n x r).
        L (cp.ndarray): Cholesky factor of the kernel matrix (r x r).        
    """
    submatrix = X[indices, :]
    V = kernel_func(X, submatrix, eta)
    A = kernel_func(submatrix, submatrix, eta)
    L = cp.linalg.cholesky(A)

    return V, L

def Nystrom_LS(X, n_components: int, kernel_func=gauss, eta=0.01, random_state=None, **kwargs):
    """
    Adaptive Nystrom sampling algorithm for kernel matrix approximation.

    Parameters:
        X (cp.ndarray): Input data matrix (n x d).
        n_components (int): Number of columns to sample.
        kernel_func: Kernel function.
        eta (float): gamma parameter of the kernel function.
        random_state: Random seed.
        kwargs: Additional arguments.

    Returns:
        V (cp.ndarray): Change of basis matrix (n x r).
        L (cp.ndarray): Cholesky factor of the kernel matrix (r x r).
        r (int): Number of columns sampled.
    """
    indices = recursiveNystrom(X, n_components, kernel_func, eta, random_state=random_state, **kwargs)
    V, L = Nystrom(X, indices, kernel_func, eta)

    return V, L

def AdaptiveNystrom(X, eta, tau):
    '''
    Adaptive Nystrom sampling algorithm for kernel matrix approximation with 
    leverage score sampling and doubling trick.

    Parameters:
        X (cp.ndarray): Input data matrix (n x d).
        eta (float): Kernel parameter.
        tau (float): Convergence tolerance parameter.

    Returns:
        V (cp.ndarray): Change of basis matrix (n x r).
        L (cp.ndarray): Cholesky factor of the kernel matrix (r x r).
        rnk (int): rank of the approximation.
    '''
    r = 1
    V = cp.zeros((X.shape[0], 1))
    L = cp.zeros((1, 1))
    while True:
        r = 2 * r  # Double the number of columns sampled

        # Compute Nystrom approximation
        V_old = V
        L_old = L
        V, L = Nystrom_LS(X, r, gauss, eta)

        # Check if LL^T is positive definite
        if cp.linalg.matrix_rank(L) < r:
            r = r // 2
            V = V_old
            L = L_old
            break

        # Compute the l-infinity component-wise error in an efficient way 
        v_i = cp.linalg.solve(L, V.T)  # Solve the linear system L v_i = V_i
        norm_i = cp.linalg.norm(v_i, axis=0)  # Compute the norm of each column of v_i
        error = 1 - cp.min(norm_i)  # Compute the error

        if error < tau:
            break
    
    rnk = cp.linalg.matrix_rank(L)
    return V, L, rnk
