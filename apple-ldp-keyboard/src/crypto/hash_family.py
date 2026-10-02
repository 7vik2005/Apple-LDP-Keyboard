import numpy as np
from typing import Union

# Field order and irreducible polynomial for GF(2^12)
FIELD_SIZE = 4096
IRREDUCIBLE_POLY = 0x1009

def _check_valid(a: Union[int, np.ndarray]):
    if isinstance(a, np.ndarray):
        if np.any((a < 0) | (a >= FIELD_SIZE)):
            raise ValueError("Elements must be in GF(2^12) [0, 4095]")
    else:
        if not (0 <= a < FIELD_SIZE):
            raise ValueError(f"Element {a} must be in GF(2^12) [0, 4095]")

def gf_add(a: Union[int, np.ndarray], b: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """Addition in GF(2^12) is bitwise XOR."""
    _check_valid(a)
    _check_valid(b)
    return a ^ b

def gf_mul(a: Union[int, np.ndarray], b: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """Multiplication in GF(2^12) modulo x^12 + x^3 + 1 (0x1009)."""
    _check_valid(a)
    _check_valid(b)
    
    res = np.zeros_like(a) if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) else 0
    temp_b = np.copy(b) if isinstance(b, np.ndarray) else b
    
    # 12 bits to process
    for i in range(12):
        # if i-th bit of a is set
        if isinstance(a, np.ndarray):
            mask = (a & (1 << i)) != 0
            res = res ^ np.where(mask, temp_b, 0)
        else:
            if (a & (1 << i)) != 0:
                res ^= temp_b
                
        # shift temp_b and reduce
        if isinstance(temp_b, np.ndarray):
            temp_b = temp_b << 1
            overflow = (temp_b & 0x1000) != 0
            temp_b = temp_b ^ np.where(overflow, IRREDUCIBLE_POLY, 0)
        else:
            temp_b <<= 1
            if temp_b & 0x1000:
                temp_b ^= IRREDUCIBLE_POLY
                
    if isinstance(res, np.ndarray):
        return res.astype(np.uint16)
    return res

def gf_square(a: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """Square an element in GF(2^12)."""
    return gf_mul(a, a)

def gf_pow(a: Union[int, np.ndarray], exponent: int) -> Union[int, np.ndarray]:
    """Exponentiation in GF(2^12)."""
    _check_valid(a)
    if exponent < 0:
        raise ValueError("Exponent must be non-negative.")
        
    res = np.ones_like(a, dtype=np.uint16) if isinstance(a, np.ndarray) else 1
    base = np.copy(a) if isinstance(a, np.ndarray) else a
    
    exp = exponent
    while exp > 0:
        if exp & 1:
            res = gf_mul(res, base)
        base = gf_square(base)
        exp >>= 1
        
    return res

def gf_inverse(a: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """Multiplicative inverse in GF(2^12). For zero, raises ValueError."""
    _check_valid(a)
    
    if isinstance(a, np.ndarray):
        if np.any(a == 0):
            raise ValueError("Cannot invert zero.")
    else:
        if a == 0:
            raise ValueError("Cannot invert zero.")
            
    # Fermat's Little Theorem in GF(2^12): a^(2^12 - 2) = a^-1
    return gf_pow(a, FIELD_SIZE - 2)


class ThreeWiseHashFamily:
    """
    Deterministic three-wise independent hash family over GF(2^12).
    Generates k hash functions evaluating P_j(x) = a_j*x^2 + b_j*x + c_j.
    """
    def __init__(self, k: int, m: int, seed: int):
        if k < 1:
            raise ValueError("k must be >= 1")
        if m <= 0 or (m & (m - 1)) != 0:
            raise ValueError("m must be a positive power of two")
        if m > 1024:
            raise ValueError("m must be <= 1024")
            
        self.k = k
        self.m = m
        self.seed = seed
        
        # We need r bits where 2^r = m
        self.mask = m - 1
        
        rng = np.random.default_rng(seed)
        
        # Sample coefficients a, b, c uniformly from GF(2^12)
        # Using uint16 for compact storage
        self._a = rng.integers(0, FIELD_SIZE, size=k, dtype=np.uint16)
        self._b = rng.integers(0, FIELD_SIZE, size=k, dtype=np.uint16)
        self._c = rng.integers(0, FIELD_SIZE, size=k, dtype=np.uint16)
        
    def hash(self, item_id: int, hash_index: int) -> int:
        """
        Evaluate the hash_index-th hash function for item_id.
        """
        if not (0 <= hash_index < self.k):
            raise ValueError(f"hash_index {hash_index} out of bounds.")
        _check_valid(item_id)
        
        a_j = int(self._a[hash_index])
        b_j = int(self._b[hash_index])
        c_j = int(self._c[hash_index])
        
        x = item_id
        x2 = gf_square(x)
        
        # a_j*x^2 + b_j*x + c_j
        term1 = gf_mul(a_j, x2)
        term2 = gf_mul(b_j, x)
        
        p_x = gf_add(gf_add(term1, term2), c_j)
        
        return int(p_x) & self.mask
        
    def hash_all(self, item_id: int) -> np.ndarray:
        """
        Evaluate all k hash functions for item_id.
        Returns a numpy array of shape (k,).
        """
        _check_valid(item_id)
        
        x = item_id
        x2 = gf_square(x)
        
        # Vectorized operations over the k coefficients
        term1 = gf_mul(self._a, x2)
        term2 = gf_mul(self._b, x)
        
        p_x = gf_add(gf_add(term1, term2), self._c)
        
        return (p_x & self.mask).astype(np.uint16)
        
    def coefficients(self):
        """
        Return a read-only copy of the internal coefficient arrays.
        """
        return (np.copy(self._a), np.copy(self._b), np.copy(self._c))
