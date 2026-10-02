import math
import numpy as np
from scipy import stats
from typing import Tuple, Union, Iterable

def mean_absolute_error(true_counts: np.ndarray, estimated_counts: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE):
    MAE = (1 / D) * sum_{i=1}^D |f_hat_i - f_i|
    """
    true_counts = np.asarray(true_counts, dtype=np.float64)
    estimated_counts = np.asarray(estimated_counts, dtype=np.float64)
    if true_counts.shape != estimated_counts.shape:
        raise ValueError("Shape mismatch between true_counts and estimated_counts.")
    if len(true_counts) == 0:
        raise ValueError("Input arrays must not be empty.")
    return float(np.mean(np.abs(estimated_counts - true_counts)))


def root_mean_square_error(true_counts: np.ndarray, estimated_counts: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE):
    RMSE = sqrt( (1 / D) * sum_{i=1}^D (f_hat_i - f_i)^2 )
    """
    true_counts = np.asarray(true_counts, dtype=np.float64)
    estimated_counts = np.asarray(estimated_counts, dtype=np.float64)
    if true_counts.shape != estimated_counts.shape:
        raise ValueError("Shape mismatch between true_counts and estimated_counts.")
    if len(true_counts) == 0:
        raise ValueError("Input arrays must not be empty.")
    return float(np.sqrt(np.mean((estimated_counts - true_counts) ** 2)))


def normalized_mae(true_counts: np.ndarray, estimated_counts: np.ndarray, total_events: int) -> float:
    """
    Calculate Normalized Mean Absolute Error (NMAE):
    NMAE = MAE / total_events
    """
    if total_events <= 0:
        raise ValueError(f"total_events must be positive, got {total_events}")
    mae = mean_absolute_error(true_counts, estimated_counts)
    return float(mae / total_events)


def top_k_overlap(true_counts: np.ndarray, estimated_counts: np.ndarray, k: int = 10, item_ids: np.ndarray = None) -> float:
    """
    Calculate Top-K Overlap ratio in [0, 1]:
    TopKOverlap = |TrueTopK intersection EstimatedTopK| / k
    
    Uses deterministic tie-breaking by item_id (ascending) if counts are tied.
    """
    true_counts = np.asarray(true_counts, dtype=np.float64)
    estimated_counts = np.asarray(estimated_counts, dtype=np.float64)
    D = len(true_counts)
    
    if D == 0:
        raise ValueError("Input arrays must not be empty.")
    
    k_actual = min(k, D)
    if k_actual <= 0:
        raise ValueError(f"k must be positive, got {k}")
        
    if item_ids is None:
        item_ids = np.arange(D)
    else:
        item_ids = np.asarray(item_ids)
        
    # Sort deterministically: descending count, ascending item_id
    # Structured array for stable lexicographical sorting
    true_struct = np.empty(D, dtype=[('count', 'f8'), ('id', 'i8')])
    true_struct['count'] = -true_counts  # negate for descending order
    true_struct['id'] = item_ids
    true_order = np.argsort(true_struct, order=['count', 'id'])
    true_top_k = set(item_ids[true_order[:k_actual]])
    
    est_struct = np.empty(D, dtype=[('count', 'f8'), ('id', 'i8')])
    est_struct['count'] = -estimated_counts  # negate for descending order
    est_struct['id'] = item_ids
    est_order = np.argsort(est_struct, order=['count', 'id'])
    est_top_k = set(item_ids[est_order[:k_actual]])
    
    overlap_count = len(true_top_k.intersection(est_top_k))
    return float(overlap_count / k_actual)


def spearman_rank_correlation(true_counts: np.ndarray, estimated_counts: np.ndarray) -> float:
    """
    Calculate Spearman rank correlation using SciPy stats.
    Returns float in [-1, 1] or NaN if mathematically undefined (e.g. constant array).
    """
    true_counts = np.asarray(true_counts, dtype=np.float64)
    estimated_counts = np.asarray(estimated_counts, dtype=np.float64)
    
    if len(true_counts) < 2:
        return float('nan')
        
    # Check for constant arrays where rank correlation is undefined
    if np.all(true_counts == true_counts[0]) or np.all(estimated_counts == estimated_counts[0]):
        return float('nan')
        
    res = stats.spearmanr(true_counts, estimated_counts)
    val = res.statistic if hasattr(res, 'statistic') else res.correlation
    
    if np.isnan(val):
        return float('nan')
    return float(val)


def cms_communication_bytes(m: int, k: int) -> int:
    """
    Calculate theoretical serialized CMS payload size per record in bytes:
    bits = m + ceil(log2(k))
    communication = ceil(bits / 8)
    """
    if m <= 0 or k <= 0:
        raise ValueError("m and k must be positive integers.")
        
    bits_hash = math.ceil(math.log2(k)) if k > 1 else 1
    total_bits = m + bits_hash
    return int(math.ceil(total_bits / 8.0))


def hcms_communication_bytes(m: int, k: int) -> int:
    """
    Calculate theoretical serialized HCMS payload size per record in bytes:
    bits = 1 + ceil(log2(k)) + ceil(log2(m))
    communication = ceil(bits / 8)
    """
    if m <= 0 or k <= 0:
        raise ValueError("m and k must be positive integers.")
        
    bits_k = math.ceil(math.log2(k)) if k > 1 else 1
    bits_m = math.ceil(math.log2(m)) if m > 1 else 1
    total_bits = 1 + bits_k + bits_m
    return int(math.ceil(total_bits / 8.0))



def cms_sketch_memory_bytes(m: int, k: int) -> Tuple[int, float]:
    """
    Calculate theoretical server sketch matrix allocation for float64 precision:
    memory_bytes = k * m * 8
    memory_mib = memory_bytes / (1024 * 1024)
    
    Returns
    -------
    (int, float)
        (sketch_memory_bytes, sketch_memory_mib)
    """
    if m <= 0 or k <= 0:
        raise ValueError("m and k must be positive integers.")
        
    bytes_val = int(k * m * 8)
    mib_val = float(bytes_val / (1024.0 * 1024.0))
    return bytes_val, mib_val
