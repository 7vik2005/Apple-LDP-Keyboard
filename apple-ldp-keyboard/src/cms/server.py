import numpy as np
from typing import Iterable, Union
from src.crypto.hash_family import ThreeWiseHashFamily
from src.cms.estimator import compute_c_epsilon, estimate_frequency, estimate_many_frequencies

class CMSServer:
    """
    Server-side Private Count Mean Sketch (CMS) aggregator and frequency estimator.
    
    Implements Apple's Private CMS Server Algorithm (Algorithm 3) and Estimator (Algorithm 4).
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
        
        # Precompute debiasing constant c_epsilon
        self.c_epsilon = compute_c_epsilon(self.epsilon)
        
        # Initialize dense sketch matrix M of shape (k, m) with float64 precision
        self.M = np.zeros((self.k, self.m), dtype=np.float64)
        
        # Total count of processed privatized records n
        self.n = 0

    def update(self, privatized_vector: np.ndarray, hash_index: int) -> None:
        """
        Process a single privatized vector and hash index, updating the sketch matrix M.
        
        Debiasing transformation (Algorithm 3):
        x_tilde = k * ( (c_epsilon / 2) * v_tilde + (1 / 2) * 1 )
        M[j, :] += x_tilde
        
        Parameters
        ----------
        privatized_vector : np.ndarray
            Privatized vector v_tilde in {-1, +1}^m.
        hash_index : int
            Selected hash function index j in {0, ..., k-1}.
        """
        if not (0 <= hash_index < self.k):
            raise ValueError(f"hash_index {hash_index} out of bounds for k={self.k}.")
            
        if not isinstance(privatized_vector, np.ndarray):
            privatized_vector = np.asarray(privatized_vector)
            
        if privatized_vector.shape != (self.m,):
            raise ValueError(f"Vector shape {privatized_vector.shape} does not match sketch width m={self.m}.")
            
        if not np.all((privatized_vector == 1) | (privatized_vector == -1)):
            raise ValueError("Privatized vector elements must be strictly -1 or +1.")
            
        # Debiasing scaling transformation
        # x_tilde = k * ( (c_epsilon / 2) * v_tilde + 0.5 )
        x_tilde = self.k * ((self.c_epsilon / 2.0) * privatized_vector + 0.5)
        
        # Update ONLY the selected row j in sketch matrix M
        self.M[hash_index, :] += x_tilde
        
        # Increment record count n
        self.n += 1

    def process(self, record: dict) -> None:
        """
        Process a client record dictionary containing 'vector' and 'hash_index'.
        
        Parameters
        ----------
        record : dict
            Dictionary returned by CMSClient.privatize().
        """
        if not isinstance(record, dict) or "vector" not in record or "hash_index" not in record:
            raise ValueError("Record must be a dict containing 'vector' and 'hash_index'.")
            
        self.update(record["vector"], record["hash_index"])

    def process_stream(self, records: Iterable[dict]) -> None:
        """
        Process an iterable stream of client privatized records.
        """
        for record in records:
            self.process(record)

    def estimate(self, item_id: int) -> float:
        """
        Estimate the raw frequency of item_id.
        
        Parameters
        ----------
        item_id : int
            Item ID in GF(2^12) [0, 4095].
            
        Returns
        -------
        float
            Unrounded, unclamped frequency estimate.
        """
        return estimate_frequency(item_id, self.M, self.epsilon, self.hash_family, self.n)

    def estimate_many(self, item_ids: Iterable[int]) -> dict:
        """
        Estimate frequencies for multiple item IDs.
        
        Parameters
        ----------
        item_ids : Iterable[int]
            Iterable of item IDs.
            
        Returns
        -------
        dict
            Mapping from item_id -> frequency estimate float.
        """
        return estimate_many_frequencies(item_ids, self.M, self.epsilon, self.hash_family, self.n)

    def reset(self) -> None:
        """
        Reset the sketch matrix M to zeros and record counter n to 0 without altering configurations.
        """
        self.M.fill(0.0)
        self.n = 0
