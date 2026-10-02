import pytest
import numpy as np
from src.privacy.randomizer import (
    randomization_probability, hcms_randomization_probability,
    randomize_sign_vector, randomize_bit
)

def test_randomization_probability_matches_theory():
    """
    Verify randomization_probability(epsilon) matches independent theoretical values
    for fixed epsilon values in {0.5, 1, 2, 4, 8} within numerical tolerance 1e-12.
    """
    # Manually defined theoretical calculations: p = exp(eps/2) / (exp(eps/2) + 1)
    epsilon_cases = {
        0.5: float(np.exp(0.25) / (np.exp(0.25) + 1.0)),
        1.0: float(np.exp(0.50) / (np.exp(0.50) + 1.0)),
        2.0: float(np.exp(1.00) / (np.exp(1.00) + 1.0)),
        4.0: float(np.exp(2.00) / (np.exp(2.00) + 1.0)),
        8.0: float(np.exp(4.00) / (np.exp(4.00) + 1.0)),
    }
    
    for eps, expected_p in epsilon_cases.items():
        actual_p = randomization_probability(eps)
        assert np.isclose(actual_p, expected_p, rtol=1e-12, atol=1e-12)

def test_hcms_randomization_probability_matches_theory():
    """
    Test 33: Verify hcms_randomization_probability(epsilon) matches independent theoretical values
    p = exp(epsilon) / (exp(epsilon) + 1) for fixed epsilon values in {0.5, 1, 2, 4, 8}.
    """
    hcms_cases = {
        0.5: float(np.exp(0.5) / (np.exp(0.5) + 1.0)),
        1.0: float(np.exp(1.0) / (np.exp(1.0) + 1.0)),
        2.0: float(np.exp(2.0) / (np.exp(2.0) + 1.0)),
        4.0: float(np.exp(4.0) / (np.exp(4.0) + 1.0)),
        8.0: float(np.exp(8.0) / (np.exp(8.0) + 1.0)),
    }
    
    for eps, expected_p in hcms_cases.items():
        actual_p = hcms_randomization_probability(eps)
        assert np.isclose(actual_p, expected_p, rtol=1e-12, atol=1e-12)

def test_cms_vs_hcms_randomization_probability_distinction():
    """
    Test 43: Regression test verifying that CMS and HCMS randomization probabilities are distinct:
    p_CMS = exp(eps/2) / (exp(eps/2) + 1) vs p_HCMS = exp(eps) / (exp(eps) + 1).
    At epsilon = 2.0: p_CMS ~ 0.731059 != p_HCMS ~ 0.880797.
    """
    eps = 2.0
    p_cms = randomization_probability(eps)
    p_hcms = hcms_randomization_probability(eps)
    
    assert not np.isclose(p_cms, p_hcms)
    assert np.isclose(p_cms, float(np.exp(1.0) / (np.exp(1.0) + 1.0)))
    assert np.isclose(p_hcms, float(np.exp(2.0) / (np.exp(2.0) + 1.0)))

def test_probability_range():
    """Verify that for all positive epsilon, 0.5 < p < 1.0 and 1 - p > 0."""
    for eps in [0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        p = randomization_probability(eps)
        p_hcms = hcms_randomization_probability(eps)
        assert 0.5 < p < 1.0
        assert 0.5 < p_hcms < 1.0
        assert 1.0 - p > 0.0
        assert 1.0 - p_hcms > 0.0

def test_randomization_input_validation():
    """Verify invalid inputs raise appropriate ValueError exceptions."""
    rng = np.random.default_rng(42)
    valid_vec = np.array([1, -1, 1, -1])
    
    # Epsilon <= 0
    with pytest.raises(ValueError):
        randomization_probability(0.0)
    with pytest.raises(ValueError):
        hcms_randomization_probability(0.0)
    with pytest.raises(ValueError):
        randomize_sign_vector(valid_vec, 0.0, rng)
    with pytest.raises(ValueError):
        randomize_bit(1, 0.0, rng)
        
    # Invalid RNG
    with pytest.raises(ValueError):
        randomize_sign_vector(valid_vec, 2.0, None)
    with pytest.raises(ValueError):
        randomize_bit(1, 2.0, None)
        
    # Invalid bit values
    with pytest.raises(ValueError):
        randomize_bit(0, 2.0, rng)

def test_hcms_randomize_bit_monte_carlo():
    """
    Test 34: Monte Carlo single-bit randomization test (100,000 trials) for epsilon=2.0.
    Compares observed sign-preservation probability with theoretical p_HCMS(eps=2) ~ 0.880797.
    """
    eps = 2.0
    expected_p = hcms_randomization_probability(eps)  # ~ 0.880797
    trials = 100000
    se = np.sqrt(expected_p * (1.0 - expected_p) / trials)  # ~ 0.00102
    tolerance = 4.0 * se  # ~ 0.004
    
    rng = np.random.default_rng(777)
    
    # Test input bit = +1
    pos_flips = [randomize_bit(1, eps, rng) for _ in range(trials)]
    observed_p_pos = np.mean(np.array(pos_flips) == 1)
    
    diff_pos = abs(observed_p_pos - expected_p)
    assert diff_pos < tolerance, f"Observed p {observed_p_pos:.5f} deviated from theoretical {expected_p:.5f} by {diff_pos:.5f}"

def test_randomization_output_properties():
    """Verify length, values, type, and non-mutation of input vector."""
    rng = np.random.default_rng(42)
    input_vec = np.array([1, -1, 1, 1, -1, -1, 1, -1])
    input_copy = input_vec.copy()
    
    out_vec = randomize_sign_vector(input_vec, epsilon=2.0, rng=rng)
    
    assert out_vec.shape == input_vec.shape
    assert np.all((out_vec == 1) | (out_vec == -1))
    np.testing.assert_array_equal(input_vec, input_copy)

def test_monte_carlo_randomization_both_signs():
    """Monte Carlo statistical test for sign-vector randomization."""
    epsilon = 2.0
    expected_p = randomization_probability(epsilon)
    trials = 200000
    se = np.sqrt(expected_p * (1.0 - expected_p) / trials)
    tolerance = 4.0 * se
    
    rng = np.random.default_rng(12345)
    
    vec_pos = np.ones(trials, dtype=np.int8)
    out_pos = randomize_sign_vector(vec_pos, epsilon=epsilon, rng=rng)
    observed_p_pos = np.mean(out_pos == 1)
    
    diff_pos = abs(observed_p_pos - expected_p)
    assert diff_pos < tolerance
    
    vec_neg = np.full(trials, -1, dtype=np.int8)
    out_neg = randomize_sign_vector(vec_neg, epsilon=epsilon, rng=rng)
    observed_p_neg = np.mean(out_neg == -1)
    
    diff_neg = abs(observed_p_neg - expected_p)
    assert diff_neg < tolerance
    assert abs(observed_p_pos - observed_p_neg) < tolerance

def test_randomization_independence_sanity_check():
    """Sanity check for coordinate randomization flip independence."""
    m = 8
    trials = 50000
    rng = np.random.default_rng(999)
    input_vec = np.tile(np.array([1, -1], dtype=np.int8), m // 2)
    
    samples = np.zeros((trials, m), dtype=np.float64)
    for i in range(trials):
        samples[i] = randomize_sign_vector(input_vec, epsilon=2.0, rng=rng)
        
    corr_matrix = np.corrcoef(samples, rowvar=False)
    np.fill_diagonal(corr_matrix, 0.0)
    max_off_diag_corr = np.max(np.abs(corr_matrix))
    assert max_off_diag_corr < 0.02

def test_empirical_privacy_sanity_check():
    """Lightweight empirical privacy sanity check."""
    from src.crypto.hash_family import ThreeWiseHashFamily
    from src.cms.client import CMSClient
    
    hash_fam = ThreeWiseHashFamily(k=16, m=32, seed=42)
    client_a = CMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(100))
    client_b = CMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(100))
    
    rec_a = client_a.privatize(item_id=10)
    rec_b = client_b.privatize(item_id=20)
    
    assert rec_a["vector"].shape == (32,)
    assert rec_b["vector"].shape == (32,)
    assert np.all((rec_a["vector"] == 1) | (rec_a["vector"] == -1))
    assert np.all((rec_b["vector"] == 1) | (rec_b["vector"] == -1))
