import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def normalize_adj_mx(adj_mx, adj_type, return_type="dense"):
    # Adjacency matrix normalization method mapping
    adj_methods = {
        "normlap": lambda x: [calculate_normalized_laplacian(x)],
        "scalap": lambda x: [calculate_scaled_laplacian(x)],
        "symadj": lambda x: [calculate_sym_adj(x)],
        "transition": lambda x: [calculate_asym_adj(x)],
        "doubletransition": lambda x: [
            calculate_asym_adj(x),
            calculate_asym_adj(np.transpose(x)),
        ],
        "identity": lambda x: [np.diag(np.ones(x.shape[0])).astype(np.float32)],
        "uqgnn": lambda x: [calculate_KGCN(x).astype(np.float32)],
    }

    if adj_type not in adj_methods:
        return []

    adj = adj_methods[adj_type](adj_mx)

    # Return format conversion
    if return_type == "dense":
        adj = [a.astype(np.float32).todense() if sp.issparse(a) else a for a in adj]
    elif return_type == "coo":
        adj = [a.tocoo() if sp.issparse(a) else sp.coo_matrix(a) for a in adj]

    return adj


def calculate_KGCN(A):
    """
    Return degree-normalized adjacency matrix (for K_GCN).

    Args:
        A: Adjacency matrix

    Returns:
        Normalized adjacency matrix
    """
    A = np.asarray(A, dtype=np.float32)

    # Add self-loops
    if A[0, 0] == 0:
        A = A + np.eye(A.shape[0], dtype=np.float32)

    # Compute degrees and normalize
    D = np.sum(A, axis=1)
    D = np.maximum(D, 1e-5)  # Avoid division by zero
    D_inv_sqrt = np.power(D, -0.5)

    # D^(-1/2) * A * D^(-1/2)
    A_wave = D_inv_sqrt[:, np.newaxis] * A * D_inv_sqrt[np.newaxis, :]
    return A_wave


def calculate_normalized_laplacian(adj_mx):
    """Compute normalized Laplacian matrix L = I - D^(-1/2)*A*D^(-1/2)"""
    adj_mx = sp.coo_matrix(adj_mx)

    # Compute degrees
    d = np.array(adj_mx.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    # Create inverse square root of degree matrix
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # Compute normalized adjacency matrix D^(-1/2)*A*D^(-1/2)
    normalized_adj = d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt).tocoo()

    # Return Laplacian matrix L = I - A
    return sp.eye(adj_mx.shape[0]) - normalized_adj


def calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=True):
    """Compute scaled Laplacian matrix ~L = (2/lambda_max)*L - I"""
    if undirected:
        adj_mx = np.maximum(adj_mx, adj_mx.T)

    L = calculate_normalized_laplacian(adj_mx)

    # Compute largest eigenvalue
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, k=1, which="LM")
        lambda_max = lambda_max[0]

    L = sp.csr_matrix(L)
    M = L.shape[0]
    I = sp.identity(M, format="csr", dtype=L.dtype)

    # Return scaled Laplacian matrix
    return (2 / lambda_max * L) - I


def calculate_sym_adj(adj_mx):
    """Compute symmetric normalized adjacency matrix A_sym = D^(-1/2)*A*D^(-1/2)"""
    adj_mx = sp.coo_matrix(adj_mx)

    # Compute out-degrees
    rowsum = np.array(adj_mx.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt)


def calculate_asym_adj(adj_mx):
    """Compute asymmetric normalized adjacency matrix A_asym = D^(-1)*A"""
    adj_mx = sp.coo_matrix(adj_mx)

    # Compute out-degrees
    rowsum = np.array(adj_mx.sum(axis=1)).flatten()
    d_inv = np.power(rowsum, -1)
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)

    return d_mat_inv.dot(adj_mx)


def calculate_cheb_poly(L, Ks):
    """Compute Chebyshev polynomial basis T_k(L) for k=0,1,...,Ks-1

    Uses recurrence relation: T_k(L) = 2*L*T_{k-1}(L) - T_{k-2}(L)

    Args:
        L: Laplacian matrix
        Ks: Polynomial order

    Returns:
        Array of Chebyshev polynomial bases [T_0(L), T_1(L), ..., T_{Ks-1}(L)]
    """
    n = L.shape[0]
    T = [np.eye(n), L.copy()]

    for k in range(2, Ks):
        T.append(2 * L @ T[k - 1] - T[k - 2])

    return np.asarray(T)
