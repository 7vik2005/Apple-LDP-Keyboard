import pytest
import numpy as np
from src.hcms.hadamard import fwht, ifwht

def test_hadamard_single_element():
    """Test 1: FWHT([1]) == [1]."""
    x = np.array([1.0])
    res = fwht(x)
    np.testing.assert_array_equal(res, np.array([1.0]))

def test_hadamard_known_small_transforms():
    """Test 2: Known small transforms computed independently."""
    # m = 2: H_2 = [[1, 1], [1, -1]] -> H_2 * [1, 2]^T = [3, -1]
    x2 = np.array([1.0, 2.0])
    res2 = fwht(x2)
    expected2 = np.array([3.0, -1.0])
    np.testing.assert_allclose(res2, expected2, rtol=1e-12)
    
    # m = 4: H_4 = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
    # H_4 * [1, 2, 3, 4]^T = [10, -2, -4, 0]
    x4 = np.array([1.0, 2.0, 3.0, 4.0])
    res4 = fwht(x4)
    expected4 = np.array([10.0, -2.0, -4.0, 0.0])
    np.testing.assert_allclose(res4, expected4, rtol=1e-12)

def test_hadamard_round_trip():
    """Test round-trip property: ifwht(fwht(x)) == x for m in {1, 2, 4, 8, 16, 32, 1024}."""
    for p in [0, 1, 2, 3, 4, 5, 10]:
        m = 1 << p
        rng = np.random.default_rng(42 + m)
        x = rng.standard_normal(m)
        
        w = fwht(x)
        x_rec = ifwht(w)
        
        np.testing.assert_allclose(x_rec, x, rtol=1e-12, atol=1e-12)

def test_hadamard_orthogonality_in_unit_test():
    """Test orthogonality property H_m * H_m^T = m * I for small dimensions using explicit matrix in test."""
    def construct_explicit_hadamard(m):
        if m == 1:
            return np.array([[1.0]])
        h_half = construct_explicit_hadamard(m // 2)
        return np.block([[h_half, h_half], [h_half, -h_half]])
        
    for m in [2, 4, 8, 16]:
        H = construct_explicit_hadamard(m)
        product = H @ H.T
        expected = float(m) * np.eye(m)
        np.testing.assert_allclose(product, expected, rtol=1e-12)

def test_hadamard_invalid_dimensions():
    """Verify non power-of-two dimensions raise ValueError."""
    with pytest.raises(ValueError):
        fwht(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        fwht(np.array([1, 2, 3, 4, 5]))
    with pytest.raises(ValueError):
        fwht(np.array([]))
