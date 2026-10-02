import numpy as np
from typing import Union, Iterable
from src.crypto.hash_family import ThreeWiseHashFamily

def compute_c_epsilon(epsilon: float) -> float:
    """
    Compute the server debiasing constant c_epsilon for Private CMS:
    c_epsilon = (exp(epsilon / 2) + 1) / (exp(epsilon / 2) - 1)
              = (1 + exp(-epsilon / 2)) / (1 - exp(-epsilon / 2))
              
    Parameters
    ----------
    epsilon : float
        Privacy parameter (epsilon > 0).
        
    Returns
    -------
    float
        Debiasing factor c_epsilon.
    """
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive (epsilon > 0), got {epsilon}")
        
    denom = 1.0 - np.exp(-epsilon / 2.0)
    if np.isclose(denom, 0.0):
        raise ValueError(f"Epsilon value {epsilon} is too small, resulting in numerical instability.")
        
    numer = 1.0 + np.exp(-epsilon / 2.0)
    return float(numer / denom)


def estimate_frequency(item_id: int, M: np.ndarray, epsilon: float, hash_family: ThreeWiseHashFamily, n: int) -> float:
    """
    Estimate the frequency of item_id using the server sketch matrix M and Apple's unbiased CMS estimator.
    
    Formula (Algorithm 4):
    f_tilde(d) = (m / (m - 1)) * [ (1 / k) * sum_{j=0}^{k-1} M[j, h_j(d)] - (n / m) ]
    
    Parameters
    ----------
    item_id : int
        The item identifier in GF(2^12) [0, 4095].
    M : np.ndarray
        The server sketch matrix of shape (k, m).
    epsilon : float
        Privacy parameter (epsilon > 0).
    hash_family : ThreeWiseHashFamily
        The 3-wise independent hash family shared between client and server.
    n : int
        Total number of privatized records processed by the server.
        
    Returns
    -------
    float
        Unrounded, unclamped frequency estimate.
    """
    if n < 0:
        raise ValueError("Record count n must be non-negative.")
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive (epsilon > 0).")
        
    k = hash_family.k
    m = hash_family.m
    
    if M.shape != (k, m):
        raise ValueError(f"Sketch matrix shape {M.shape} does not match hash family parameters (k={k}, m={m}).")
        
    # Evaluate all k hash functions for item_id
    # h_indices is an array of length k containing bucket indices in [0, m-1]
    h_indices = hash_family.hash_all(item_id)
    
    # Extract M[j, h_j(item_id)] for j=0..k-1
    # Advanced indexing: row indices 0..k-1, column indices h_indices
    row_indices = np.arange(k)
    sketch_sum = np.sum(M[row_indices, h_indices])
    
    # Unbiased estimate calculation
    raw_mean = sketch_sum / k
    unbiased = (m / (m - 1.0)) * (raw_mean - (n / m))
    
    return float(unbiased)


def estimate_many_frequencies(item_ids: Iterable[int], M: np.ndarray, epsilon: float, hash_family: ThreeWiseHashFamily, n: int) -> dict:
    """
    Estimate frequencies for a list of item IDs.
    
    Returns
    -------
    dict
        Mapping of item_id -> estimated_frequency float.
    """
    return {item_id: estimate_frequency(item_id, M, epsilon, hash_family, n) for item_id in item_ids}
