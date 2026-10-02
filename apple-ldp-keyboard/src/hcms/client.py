import numpy as np
from src.crypto.hash_family import ThreeWiseHashFamily, _check_valid
from src.privacy.randomizer import randomize_bit
from src.hcms.hadamard import fwht

class HCMSClient:
    """
    Client-side Private Hadamard Count Mean Sketch (HCMS) encoder and randomizer under local differential privacy.
    
    Implements Apple's Private HCMS Client Algorithm (Algorithm 6).
    Transmits only ONE privatized bit + hash index j + Hadamard coordinate index l per event.
    """
    def __init__(self, epsilon: float, hash_family: ThreeWiseHashFamily, rng: np.random.Generator):
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive (epsilon > 0), got {epsilon}")
        if hash_family is None or not isinstance(hash_family, ThreeWiseHashFamily):
            raise ValueError("A valid ThreeWiseHashFamily instance must be provided.")
        if rng is None or not isinstance(rng, np.random.Generator):
            raise ValueError("A valid np.random.Generator instance (rng) must be provided.")
            
        self.epsilon = float(epsilon)
        self.hash_family = hash_family
        self.rng = rng
        
    def privatize(self, item_id: int) -> dict:
        """
        Privatize an item ID according to Apple's Private HCMS client algorithm (Algorithm 6).
        
        Steps:
        1. Sample hash function index j ~ Uniform({0, ..., k-1}).
        2. Construct one-hot vector v in {0, 1}^m with v[h_j(item_id)] = 1 and 0 elsewhere.
        3. Compute Hadamard transform w = H_m * v via Fast Walsh-Hadamard Transform (FWHT).
        4. Sample coordinate index l ~ Uniform({0, ..., m-1}).
        5. Extract scalar Hadamard bit w_l in {-1, +1}.
        6. Apply single-bit local LDP randomization: privatized_bit = randomize_bit(w_l, epsilon, rng).
        7. Return payload containing ONLY privatized bit, hash_index j, and coordinate_index l.
        
        Parameters
        ----------
        item_id : int
            Input item ID in GF(2^12) [0, 4095].
            
        Returns
        -------
        dict
            Immutable payload: {"bit": privatized_bit, "hash_index": j, "coordinate_index": l}
        """
        _check_valid(item_id)
        
        k = self.hash_family.k
        m = self.hash_family.m
        
        # Step 1: Uniformly sample hash index j in {0, ..., k-1}
        j = int(self.rng.integers(0, k))
        
        # Step 2: Evaluate hash value h_j(item_id) and construct one-hot vector v with v[h_val] = 1, others 0 (Algorithm 6)
        h_val = self.hash_family.hash(item_id, j)
        v = np.zeros(m, dtype=np.float64)
        v[h_val] = 1.0
        
        # Step 3: Compute Fast Walsh-Hadamard Transform w = H_m * v
        w = fwht(v)
        
        # Step 4: Uniformly sample coordinate index l in {0, ..., m-1}
        l = int(self.rng.integers(0, m))
        
        # Step 5: Extract bit w_l (since v is one-hot, w_l = (H_m)_{l, h_val} is strictly in {-1, +1})
        w_l = int(round(w[l]))
        
        # Step 6: Apply single-bit local LDP randomization (probability p = exp(eps)/(exp(eps)+1))
        privatized_bit = randomize_bit(w_l, self.epsilon, self.rng)
        
        # Step 7: Return compact payload without raw item identifier
        return {
            "bit": int(privatized_bit),
            "hash_index": j,
            "coordinate_index": l
        }
