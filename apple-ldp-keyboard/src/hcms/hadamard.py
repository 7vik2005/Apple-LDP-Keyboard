import numpy as np
import math

def fwht(vector: np.ndarray) -> np.ndarray:
    """
    Fast Walsh-Hadamard Transform (FWHT) for vectors of length m = 2^p.
    Computes unnormalized forward transform in O(m log m) time.
    
    Formula:
    w = H_m * v, where H_m H_m^T = m I.
    
    Parameters
    ----------
    vector : np.ndarray
        Input 1D array of length m (power of 2).
        
    Returns
    -------
    np.ndarray
        Unnormalized Hadamard transform output array of float64 dtype.
    """
    if not isinstance(vector, np.ndarray):
        a = np.asarray(vector, dtype=np.float64)
    else:
        a = vector.astype(np.float64, copy=True)
        
    if a.ndim != 1:
        raise ValueError(f"Input vector must be 1-dimensional, got shape {a.shape}")
        
    m = a.shape[0]
    if m <= 0 or (m & (m - 1)) != 0:
        raise ValueError(f"Vector length m must be a positive power of 2, got {m}")
        
    # Vectorized Fast Walsh-Hadamard Transform
    h = 1
    while h < m:
        a = a.reshape(-1, 2, h)
        x = a[:, 0, :].copy()
        y = a[:, 1, :].copy()
        a[:, 0, :] = x + y
        a[:, 1, :] = x - y
        h *= 2
        
    return a.reshape(-1)



def ifwht(vector: np.ndarray) -> np.ndarray:
    """
    Inverse Fast Walsh-Hadamard Transform for vectors of length m = 2^p.
    Computes: ifwht(w) = fwht(w) / m.
    
    Parameters
    ----------
    vector : np.ndarray
        Input 1D array of length m (power of 2).
        
    Returns
    -------
    np.ndarray
        Inverse Hadamard transform array of float64 dtype.
    """
    if not isinstance(vector, np.ndarray):
        a = np.asarray(vector, dtype=np.float64)
    else:
        a = vector.astype(np.float64, copy=True)
        
    m = a.shape[0]
    if m <= 0 or (m & (m - 1)) != 0:
        raise ValueError(f"Vector length m must be a positive power of 2, got {m}")
        
    return fwht(a) / float(m)
