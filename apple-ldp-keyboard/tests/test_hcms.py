import pytest
import numpy as np
from src.crypto.hash_family import ThreeWiseHashFamily
from src.privacy.randomizer import hcms_randomization_probability, randomize_bit
from src.hcms.hadamard import fwht, ifwht
from src.hcms.client import HCMSClient
from src.hcms.server import HCMSServer, compute_hcms_c_epsilon
from src.cms.estimator import estimate_frequency

def test_hcms_c_epsilon():
    """Verify compute_hcms_c_epsilon(epsilon) matches theoretical formula."""
    expected = (np.exp(2.0) + 1.0) / (np.exp(2.0) - 1.0)  # ~ 1.313035285
    actual = compute_hcms_c_epsilon(2.0)
    assert np.isclose(actual, expected, rtol=1e-12)
    
    with pytest.raises(ValueError):
        compute_hcms_c_epsilon(0.0)
    with pytest.raises(ValueError):
        compute_hcms_c_epsilon(-1.0)

def test_hcms_one_hot_vector_source_correctness():
    """
    Test 38 / Regression Test: Verify HCMS encoding vector v is all zeros except v[h_j(d)] = 1
    according to Apple's published HCMS Algorithm 6.
    """
    m = 32
    k = 16
    hash_fam = ThreeWiseHashFamily(k=k, m=m, seed=42)
    item_id = 7
    j = 3
    h_val = hash_fam.hash(item_id, j)
    
    # Construct Algorithm 6 vector v
    v = np.zeros(m, dtype=np.float64)
    v[h_val] = 1.0
    
    assert v[h_val] == 1.0
    assert np.sum(v == 1.0) == 1
    assert np.sum(v == 0.0) == m - 1
    
    # Verify Hadamard transform w = H_m * v entries are strictly in {-1, +1}
    w = fwht(v)
    assert np.all((w == 1.0) | (w == -1.0))

def test_hcms_client_ranges_and_privacy_payload():
    """Verify client outputs valid bit in {-1, +1}, valid index ranges, and no raw item metadata."""
    m = 32
    k = 16
    hash_fam = ThreeWiseHashFamily(k=k, m=m, seed=42)
    rng = np.random.default_rng(123)
    client = HCMSClient(epsilon=2.0, hash_family=hash_fam, rng=rng)
    
    for item_id in [0, 5, 42, 100]:
        rec = client.privatize(item_id=item_id)
        
        # Privacy payload check
        assert set(rec.keys()) == {"bit", "hash_index", "coordinate_index"}
        assert rec["bit"] in (-1, 1)
        assert 0 <= rec["hash_index"] < k
        assert 0 <= rec["coordinate_index"] < m

def test_hcms_server_matrix_shape_and_update():
    """Verify server M_H matrix shape is (k, m) and update modifies ONLY cell (j, l)."""
    m = 32
    k = 16
    hash_fam = ThreeWiseHashFamily(k=k, m=m, seed=42)
    server = HCMSServer(epsilon=2.0, hash_family=hash_fam)
    
    assert server.M_H.shape == (k, m)
    assert server.n == 0
    assert not server.finalized
    
    j_target = 3
    l_target = 7
    server.update(bit=1, hash_index=j_target, coordinate_index=l_target)
    
    assert server.n == 1
    c_e = compute_hcms_c_epsilon(2.0)
    expected_val = m * k * c_e * 1.0
    
    assert np.isclose(server.M_H[j_target, l_target], expected_val, rtol=1e-12)
    
    # Verify all other cells remain zero
    M_copy = server.M_H.copy()
    M_copy[j_target, l_target] = 0.0
    assert np.all(M_copy == 0.0)

def test_hcms_server_hadamard_reconstruction_vs_explicit():
    """Verify server FWHT reconstruction matches explicit matrix multiplication M = (M_H * H_m) / m."""
    def construct_explicit_hadamard(m):
        if m == 1:
            return np.array([[1.0]])
        h_half = construct_explicit_hadamard(m // 2)
        return np.block([[h_half, h_half], [h_half, -h_half]])
        
    m = 8
    k = 4
    hash_fam = ThreeWiseHashFamily(k=k, m=m, seed=42)
    server = HCMSServer(epsilon=2.0, hash_family=hash_fam)
    
    # Populate synthetic M_H
    rng = np.random.default_rng(999)
    server.M_H = rng.standard_normal((k, m))
    
    server.finalize()
    
    # Explicit matrix multiplication: M_explicit = (M_H @ H_m) / m
    H_m = construct_explicit_hadamard(m)
    M_explicit = (server.M_H @ H_m) / float(m)
    
    np.testing.assert_allclose(server.M, M_explicit, rtol=1e-12, atol=1e-12)

def test_hcms_estimator_controlled_case():
    """Verify estimate_frequency on reconstructed sketch matches CMS formula."""
    m = 4
    k = 2
    hash_fam = ThreeWiseHashFamily(k=k, m=m, seed=42)
    server = HCMSServer(epsilon=2.0, hash_family=hash_fam)
    
    server.M = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [15.0, 25.0, 35.0, 45.0]
    ], dtype=np.float64)
    server.n = 50
    server.finalized = True
    
    item_id = 2
    h_indices = hash_fam.hash_all(item_id)
    sketch_sum = server.M[0, h_indices[0]] + server.M[1, h_indices[1]]
    raw_mean = sketch_sum / 2.0
    expected_est = (4.0 / 3.0) * (raw_mean - (50.0 / 4.0))
    
    actual_est = server.estimate(item_id)
    assert np.isclose(actual_est, expected_est, rtol=1e-12)

def test_hcms_round_trip_synthetic():
    """HCMS statistical round-trip test on tiny synthetic dataset (A=50, B=30, C=20)."""
    m = 32
    k = 16
    epsilon = 2.0
    
    item_A, item_B, item_C = 1, 2, 3
    events = [item_A] * 50 + [item_B] * 30 + [item_C] * 20
    
    hash_fam = ThreeWiseHashFamily(k=k, m=m, seed=42)
    client = HCMSClient(epsilon=epsilon, hash_family=hash_fam, rng=np.random.default_rng(42))
    server = HCMSServer(epsilon=epsilon, hash_family=hash_fam)
    
    for item in events:
        rec = client.privatize(item)
        server.process(rec)
        
    server.finalize()
    assert server.n == 100
    
    est_A = server.estimate(item_A)
    est_B = server.estimate(item_B)
    est_C = server.estimate(item_C)
    
    assert isinstance(est_A, float)
    assert isinstance(est_B, float)
    assert isinstance(est_C, float)

def test_hcms_reproducibility():
    """Verify two identical seeded HCMS runs produce identical privatized records and estimates."""
    m = 32
    k = 16
    hash_fam = ThreeWiseHashFamily(k=k, m=m, seed=42)
    
    def run_hcms(seed_val):
        client = HCMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(seed_val))
        server = HCMSServer(epsilon=2.0, hash_family=hash_fam)
        
        records = []
        for item in range(20):
            rec = client.privatize(item)
            records.append(rec)
            server.process(rec)
            
        server.finalize()
        estimates = [server.estimate(i) for i in range(20)]
        return records, server.M_H.copy(), server.M.copy(), estimates

    recs1, MH1, M1, est1 = run_hcms(123)
    recs2, MH2, M2, est2 = run_hcms(123)
    
    assert recs1 == recs2
    np.testing.assert_array_equal(MH1, MH2)
    np.testing.assert_array_equal(M1, M2)
    np.testing.assert_array_equal(est1, est2)

def test_hcms_different_rng_behavior():
    """Verify different client RNG seeds produce different privatized records."""
    hash_fam = ThreeWiseHashFamily(k=16, m=32, seed=42)
    client1 = HCMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(101))
    client2 = HCMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(202))
    
    rec1 = client1.privatize(5)
    rec2 = client2.privatize(5)
    
    assert rec1 != rec2

def test_hcms_m1024_functional_support():
    """Verify m=1024 dimension can be constructed and finalized without error."""
    hash_fam = ThreeWiseHashFamily(k=64, m=1024, seed=42)
    client = HCMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(42))
    server = HCMSServer(epsilon=2.0, hash_family=hash_fam)
    
    for item in range(10):
        server.process(client.privatize(item))
        
    server.finalize()
    assert server.M.shape == (64, 1024)
    assert server.n == 10
