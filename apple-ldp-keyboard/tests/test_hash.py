import pytest
import numpy as np
from src.crypto.hash_family import (
    gf_add, gf_mul, gf_square, gf_pow, gf_inverse, 
    ThreeWiseHashFamily, FIELD_SIZE
)

def test_field_addition_properties():
    """Verify additive identity and self-inverse in GF(2^12)."""
    for a in [0, 1, 123, 456, 4095]:
        assert gf_add(a, 0) == a
        assert gf_add(0, a) == a
        assert gf_add(a, a) == 0
        assert gf_add(a, 456) == a ^ 456

def test_field_multiplication_properties():
    """Verify multiplicative identity, zero annihilation, and commutativity in GF(2^12)."""
    for a in [0, 1, 123, 456, 4095]:
        assert gf_mul(a, 0) == 0
        assert gf_mul(0, a) == 0
        assert gf_mul(a, 1) == a
        assert gf_mul(1, a) == a
        for b in [0, 1, 99, 2048, 4095]:
            assert gf_mul(a, b) == gf_mul(b, a)

def test_field_distributivity():
    """Verify a * (b + c) == (a * b) + (a * c) in GF(2^12)."""
    for a in [7, 42, 1024]:
        for b in [12, 99, 2000]:
            for c in [3, 400, 3999]:
                lhs = gf_mul(a, gf_add(b, c))
                rhs = gf_add(gf_mul(a, b), gf_mul(a, c))
                assert lhs == rhs

def test_field_closure():
    """Verify operations stay within [0, FIELD_SIZE - 1]."""
    for a in [0, 1, 42, 1024, 4095]:
        for b in [0, 1, 99, 2048, 4095]:
            prod = gf_mul(a, b)
            assert 0 <= prod < FIELD_SIZE
            
    for a in [0, 1, 7, 512, 4000]:
        for n in [0, 1, 2, 10]:
            p = gf_pow(a, n)
            assert 0 <= p < FIELD_SIZE

def test_field_inverse():
    """Verify multiplicative inverse property a * a^-1 = 1."""
    for a in [1, 2, 3, 42, 1024, 4095]:
        inv = gf_inverse(a)
        assert gf_mul(a, inv) == 1
        
    with pytest.raises(ValueError):
        gf_inverse(0)

def test_hash_output_range():
    """Test 3: Verify 0 <= h(item) < m for all m."""
    for m in [32, 128, 1024]:
        h = ThreeWiseHashFamily(k=16, m=m, seed=42)
        for i in range(16):
            output = h.hash(42, i)
            assert 0 <= output < m

def test_deterministic_hashing():
    """Test 1: Verify same family and same seed produce identical coefficients."""
    H1 = ThreeWiseHashFamily(128, 128, 42)
    H2 = ThreeWiseHashFamily(128, 128, 42)
    
    a1, b1, c1 = H1.coefficients()
    a2, b2, c2 = H2.coefficients()
    
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)
    np.testing.assert_array_equal(c1, c2)
    
    for i in range(128):
        assert H1.hash(123, i) == H2.hash(123, i)
        
    np.testing.assert_array_equal(H1.hash_all(123), H2.hash_all(123))

def test_same_item_consistency():
    """Test 2: Same item and hash index always yields identical hash output."""
    H = ThreeWiseHashFamily(k=16, m=128, seed=99)
    for item_id in [0, 1, 42, 100, 4095]:
        for hash_idx in range(16):
            out1 = H.hash(item_id, hash_idx)
            out2 = H.hash(item_id, hash_idx)
            assert out1 == out2

def test_client_server_hash_family_consistency():
    """Test 4: Client-side and server-side hash families instantiated separately agree."""
    client_hashes = ThreeWiseHashFamily(k=64, m=128, seed=42)
    server_hashes = ThreeWiseHashFamily(k=64, m=128, seed=42)
    
    for item_id in range(50):
        np.testing.assert_array_equal(client_hashes.hash_all(item_id), server_hashes.hash_all(item_id))

def test_different_seeds():
    """Verify different seeds produce different coefficients."""
    H1 = ThreeWiseHashFamily(128, 128, 42)
    H2 = ThreeWiseHashFamily(128, 128, 43)
    
    a1, b1, c1 = H1.coefficients()
    a2, b2, c2 = H2.coefficients()
    
    assert not np.array_equal(a1, a2)

def test_hash_all():
    """Verify vectorized hash_all matches individual hash calls."""
    H = ThreeWiseHashFamily(k=128, m=128, seed=42)
    item_id = 999
    
    out = H.hash_all(item_id)
    assert out.shape == (128,)
    assert np.issubdtype(out.dtype, np.integer)
    assert (out >= 0).all() and (out < 128).all()
    
    for j in [0, 10, 64, 127]:
        assert out[j] == H.hash(item_id, j)

def test_three_wise_algebraic_property():
    """Verify Vandermonde-style determinacy for 3 distinct points in GF(2^12)."""
    x1, x2, x3 = 10, 20, 30
    
    x1_sq = gf_square(x1)
    x2_sq = gf_square(x2)
    x3_sq = gf_square(x3)
    
    term1 = gf_mul(x1_sq, gf_add(x2, x3))
    term2 = gf_mul(x1, gf_add(x2_sq, x3_sq))
    term3 = gf_add(gf_mul(x2_sq, x3), gf_mul(x3_sq, x2))
    
    det = gf_add(gf_add(term1, term2), term3)
    assert det != 0

def test_different_item_ids():
    """Verify hashing across entire 2600 dictionary range produces valid outputs."""
    H = ThreeWiseHashFamily(k=16, m=1024, seed=42)
    
    for i in range(2600):
        out = H.hash(i, 0)
        assert 0 <= out < 1024
