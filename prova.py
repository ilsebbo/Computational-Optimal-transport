from utils import *
import numpy as np

X = np.random.rand(10, 2)
Y = np.random.rand(5, 2)

S = recursiveNystrom(X,3)
# Compute Nystrom approximation

d = 2
n = 10
r = 24

X = np.random.rand(n, 5)
V, L, rnk = AdaptiveNystrom(X, 0.1, 0.1)

print(f"V shape: {V.shape}")
print(f"L shape: {L.shape}")
print(f"Rank: {rnk}")

K = gauss(X, X, 0.1)
invL = np.linalg.inv(L)
K_approx = V @ invL.T @ invL @ V.T


# define p and q probability vectors
p = np.random.rand(n)
p = p / np.sum(p)
q = np.random.rand(n)
q = q / np.sum(q)

print(p)
D1, D2, W_hat = sinkhorn(V, L, p, q, 1)


