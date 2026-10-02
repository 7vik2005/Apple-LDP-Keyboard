import numpy as np
from typing import Iterable, Union
from src.crypto.hash_family import ThreeWiseHashFamily
from src.cms.estimator import estimate_frequency, estimate_many_frequencies
from src.hcms.hadamard import fwht

def compute_hcms_c_epsilon(epsilon: float) -> float:
    """
    Compute the server debiasing constant c_epsilon for Private HCMS:
    c_epsilon = (exp(epsilon) + 1) / (exp(epsilon) - 1)
              = (1 + exp(-epsilon)) / (1 - exp(-epsilon))
              
    Parameters
    ----------
    epsilon : float
        Privacy parameter (epsilon > 0).
        
    Returns
    -------
    float
        Debiasing factor c_epsilon for HCMS.
    """
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive (epsilon > 0), got {epsilon}")
        
    denom = 1.0 - np.exp(-epsilon)
    if np.isclose(denom, 0.0):
        raise ValueError(f"Epsilon value {epsilon} is too small, resulting in numerical instability.")
        
    numer = 1.0 + np.exp(-epsilon)
    return float(numer / denom)


class HCMSServer:
    """
    Server-side Private Hadamard Count Mean Sketch (HCMS) aggregator and frequency estimator.
    
    Implements Apple's Private HCMS Server Algorithm (Algorithm 7).
    Receives single privatized bits, accumulates a sparse sketch matrix M_H, reconstructs
    the sketch matrix M via inverse Fast Walsh-Hadamard Transform (FWHT), and evaluates frequency estimates.
    """
    def __init__(self, epsilon: float, hash_family: ThreeWiseHashFamily):
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive (epsilon > 0), got {epsilon}")
        if hash_family is None or not isinstance(hash_family, ThreeWiseHashFamily):
            raise ValueError("A valid ThreeWiseHashFamily instance must be provided.")
            
        self.epsilon = float(epsilon)
        self.hash_family = hash_family
        self.k = hash_family.k
        self.m = hash_family.m
        
        # Precompute HCMS debiasing constant c_epsilon
        self.c_epsilon = compute_hcms_c_epsilon(self.epsilon)
        
        # Initialize sparse sketch matrix M_H of shape (k, m) with float64 precision
        self.M_H = np.zeros((self.k, self.m), dtype=np.float64)
        
        # Reconstructed sketch matrix M of shape (k, m), computed during finalize()
        self.M = None
        
        # Total count of processed privatized records n
        self.n = 0
        
        # Lifecycle state flag
        self.finalized = False

    def update(self, bit: int, hash_index: int, coordinate_index: int) -> None:
        """
        Process a single privatized bit, hash index j, and coordinate index l.
        Updates ONLY cell M_H[j, l] (Algorithm 7).
        
        Parameters
        ----------
        bit : int
            Privatized scalar bit in {-1, +1}.
        hash_index : int
            Hash function index j in {0, ..., k-1}.
        coordinate_index : int
            Hadamard coordinate index l in {0, ..., m-1}.
        """
        if not (0 <= hash_index < self.k):
            raise ValueError(f"hash_index {hash_index} out of bounds for k={self.k}.")
        if not (0 <= coordinate_index < self.m):
            raise ValueError(f"coordinate_index {coordinate_index} out of bounds for m={self.m}.")
        if bit not in (-1, 1):
            raise ValueError(f"Privatized bit must be strictly -1 or +1, got {bit}")
            
        # Single-cell update: x_tilde = m * k * c_epsilon * bit
        x_tilde = self.m * self.k * self.c_epsilon * float(bit)
        
        # Update ONLY cell M_H[j, l]
        self.M_H[hash_index, coordinate_index] += x_tilde
        
        # Increment record count n and reset finalized state flag
        self.n += 1
        self.finalized = False

    def process(self, record: dict) -> None:
        """
        Process a client record dictionary containing 'bit', 'hash_index', and 'coordinate_index'.
        """
        if not isinstance(record, dict) or "bit" not in record or "hash_index" not in record or "coordinate_index" not in record:
            raise ValueError("Record must be a dict containing 'bit', 'hash_index', and 'coordinate_index'.")
            
        self.update(record["bit"], record["hash_index"], record["coordinate_index"])

    def process_stream(self, records: Iterable[dict]) -> None:
        """
        Process an iterable stream of HCMS client records.
        """
        for record in records:
            self.process(record)

    def finalize(self) -> None:
        """
        Reconstruct the full sketch matrix M from sparse sketch M_H via row-wise Hadamard transform.
        Formula (Algorithm 7): M = (M_H * H_m^T) / m
        
        Uses Fast Walsh-Hadamard Transform (FWHT) without constructing explicit m x m matrices.
        """
        self.M = np.zeros((self.k, self.m), dtype=np.float64)
        
        for j in range(self.k):
            # Apply unnormalized FWHT to row j and divide by m: FWHT(row) / m == row * H_m / m
            self.M[j, :] = fwht(self.M_H[j, :]) / float(self.m)
            
        self.finalized = True

    def estimate(self, item_id: int) -> float:
        """
        Estimate the frequency of item_id using the reconstructed sketch matrix M and unbiased estimator.
        Auto-finalizes if not already finalized.
        """
        if not self.finalized or self.M is None:
            self.finalize()
            
        return estimate_frequency(item_id, self.M, self.epsilon, self.hash_family, self.n)

    def estimate_many(self, item_ids: Iterable[int]) -> dict:
        """
        Estimate frequencies for multiple item IDs.
        Auto-finalizes once before evaluating estimates.
        """
        if not self.finalized or self.M is None:
            self.finalize()
            
        return estimate_many_frequencies(item_ids, self.M, self.epsilon, self.hash_family, self.n)

    def reset(self) -> None:
        """
        Reset sparse sketch matrix M_H to zeros, clear reconstructed matrix M, and reset record counter n to 0.
        """
        self.M_H.fill(0.0)
        self.M = None
        self.n = 0
        self.finalized = False
