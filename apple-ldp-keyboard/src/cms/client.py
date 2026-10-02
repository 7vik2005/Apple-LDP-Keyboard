import numpy as np
from src.crypto.hash_family import ThreeWiseHashFamily, _check_valid
from src.privacy.randomizer import randomize_sign_vector

class CMSClient:
    """
    Client-side Private Count Mean Sketch (CMS) encoder and randomizer under local differential privacy.
    
    Implements Apple's Private CMS Client Algorithm (Algorithm 2).
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
        Privatize an item ID according to Apple's Private CMS algorithm.
        
        Steps:
        1. Select hash function index j ~ Uniform({0, ..., k-1}) using explicit RNG.
        2. Construct sign encoding vector v in {-1, +1}^m with +1 at position h_j(item_id) and -1 elsewhere.
        3. Apply independent local sign randomization to v using privacy parameter epsilon.
        4. Return privatized record containing ONLY privatized_vector and hash_index j.
        
        Parameters
        ----------
        item_id : int
            Input item ID in GF(2^12) [0, 4095].
            
        Returns
        -------
        dict
            Immutable-style record: {"vector": privatized_vector, "hash_index": j}
        """
        _check_valid(item_id)
        
        k = self.hash_family.k
        m = self.hash_family.m
        
        # Step 1: Uniformly sample hash index j in {0, ..., k-1}
        # Explicit NumPy generator ensures reproducible client sampling stream
        j = int(self.rng.integers(0, k))
        
        # Step 2: Compute hash value h_j(item_id) in {0, ..., m-1}
        h_val = self.hash_family.hash(item_id, j)
        
        # Construct encoding vector v in {-1, +1}^m: v[h_val] = +1, others -1
        v = np.full(m, -1, dtype=np.int8)
        v[h_val] = 1
        
        # Step 3: Apply local epsilon-LDP sign randomization
        privatized_v = randomize_sign_vector(v, self.epsilon, self.rng)
        
        # Step 4: Return privatized payload without raw item identifier
        return {
            "vector": privatized_v,
            "hash_index": j
        }
