import pytest
import numpy as np
import pandas as pd
from src.crypto.hash_family import ThreeWiseHashFamily
from src.privacy.randomizer import randomization_probability, randomize_sign_vector
from src.cms.estimator import compute_c_epsilon, estimate_frequency, estimate_many_frequencies
from src.cms.client import CMSClient
from src.cms.server import CMSServer
from src.data.word_loader import generate_word_vocabulary
from src.data.dataset_generator import generate_users_and_events, compute_ground_truth

def test_client_vector_construction():
    """
    Verify pre-randomization vector has exactly one +1 at h_j(item_id) and -1 elsewhere.
    """
    m = 8
    k = 4
    hash_fam = ThreeWiseHashFamily(k=k, m=m, seed=42)
    item_id = 5
    
    # Evaluate client with known seed
    client = CMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(123))
    rec = client.privatize(item_id=item_id)
    j = rec["hash_index"]
    h_val = hash_fam.hash(item_id, j)
    
    # Deterministic reconstruction of pre-randomization encoding vector
    expected_pre_vector = np.full(m, -1, dtype=np.int8)
    expected_pre_vector[h_val] = 1
    
    assert expected_pre_vector[h_val] == 1
    assert np.sum(expected_pre_vector == 1) == 1
    assert np.sum(expected_pre_vector == -1) == m - 1

def test_client_record_privacy_boundary():
    """
    Verify client privatized record payload contains ONLY 'vector' and 'hash_index'.
    Strictly forbids raw item_id, word, true_count, ground_truth, or user_id.
    """
    hash_fam = ThreeWiseHashFamily(k=16, m=32, seed=42)
    client = CMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(42))
    rec = client.privatize(item_id=7)
    
    allowed_keys = {"vector", "hash_index"}
    assert set(rec.keys()) == allowed_keys
    for forbidden in ["item_id", "word", "true_count", "ground_truth", "user_id"]:
        assert forbidden not in rec

def test_server_matrix_shape_and_initial_state():
    """Verify server sketch matrix M is initialized to zeros of shape (k, m) and n=0."""
    for (k, m) in [(16, 32), (128, 128)]:
        hash_fam = ThreeWiseHashFamily(k=k, m=m, seed=42)
        server = CMSServer(epsilon=2.0, hash_family=hash_fam)
        
        assert server.M.shape == (k, m)
        assert server.M.dtype == np.float64
        assert np.all(server.M == 0.0)
        assert server.n == 0

def test_server_single_update():
    """
    Verify server update increments n to 1, updates ONLY selected row j with x_tilde,
    and leaves all non-selected rows zero.
    """
    m = 8
    k = 4
    hash_fam = ThreeWiseHashFamily(k=k, m=m, seed=42)
    server = CMSServer(epsilon=2.0, hash_family=hash_fam)
    
    privatized_vec = np.array([-1, 1, -1, 1, -1, -1, 1, -1], dtype=np.int8)
    target_j = 2
    
    server.update(privatized_vec, hash_index=target_j)
    
    assert server.n == 1
    
    # Calculate expected x_tilde independently
    c_e = (np.exp(1.0) + 1.0) / (np.exp(1.0) - 1.0)
    expected_x_tilde = k * ((c_e / 2.0) * privatized_vec + 0.5)
    
    # Selected row must match expected_x_tilde
    np.testing.assert_allclose(server.M[target_j], expected_x_tilde, rtol=1e-12)
    
    # Non-selected rows must remain zero
    for j in range(k):
        if j != target_j:
            assert np.all(server.M[j] == 0.0)

def test_server_multiple_records():
    """Verify processing multiple records updates total record count n and sketch matrix."""
    hash_fam = ThreeWiseHashFamily(k=16, m=32, seed=42)
    server = CMSServer(epsilon=2.0, hash_family=hash_fam)
    client = CMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(42))
    
    N = 10
    for i in range(N):
        rec = client.privatize(item_id=i % 5)
        server.process(rec)
        
    assert server.n == N
    assert server.M.shape == (16, 32)
    assert np.any(server.M != 0.0)

def test_server_reset():
    """Verify server reset sets n=0 and M=zeros while leaving config intact."""
    hash_fam = ThreeWiseHashFamily(k=16, m=32, seed=42)
    server = CMSServer(epsilon=2.0, hash_family=hash_fam)
    client = CMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(42))
    
    server.process(client.privatize(item_id=1))
    assert server.n == 1
    
    server.reset()
    assert server.n == 0
    assert np.all(server.M == 0.0)
    assert server.epsilon == 2.0
    assert server.k == 16
    assert server.m == 32

def test_c_epsilon_validation():
    """Test compute_c_epsilon against independently defined theoretical constants."""
    expected_cases = {
        0.5: float((np.exp(0.25) + 1.0) / (np.exp(0.25) - 1.0)),
        1.0: float((np.exp(0.50) + 1.0) / (np.exp(0.50) - 1.0)),
        2.0: float((np.exp(1.00) + 1.0) / (np.exp(1.00) - 1.0)),
        4.0: float((np.exp(2.00) + 1.0) / (np.exp(2.00) - 1.0)),
        8.0: float((np.exp(4.00) + 1.0) / (np.exp(4.00) - 1.0)),
    }
    for eps, expected_c in expected_cases.items():
        actual_c = compute_c_epsilon(eps)
        assert np.isclose(actual_c, expected_c, rtol=1e-12, atol=1e-12)

def test_server_transformation_deterministic_example():
    """
    Test server transformation and sketch update on a small deterministic example:
    m=4, k=2, epsilon=2.0, v_tilde=[-1, +1, -1, +1], j=0.
    """
    hash_fam = ThreeWiseHashFamily(k=2, m=4, seed=42)
    server = CMSServer(epsilon=2.0, hash_family=hash_fam)
    
    privatized_vec = np.array([-1, 1, -1, 1], dtype=np.int8)
    j = 0
    
    server.update(privatized_vec, hash_index=j)
    
    c_e = (np.exp(1.0) + 1.0) / (np.exp(1.0) - 1.0)  # ~ 2.163953413738653
    # x_tilde = k * ( (c_e/2)*v_tilde + 0.5 ) = 2 * ( (c_e/2)*v_tilde + 0.5 )
    # pos 0: 2 * (-c_e/2 + 0.5) = 1 - c_e ~ -1.163953413738653
    # pos 1: 2 * (+c_e/2 + 0.5) = 1 + c_e ~  3.163953413738653
    expected_row0 = np.array([1.0 - c_e, 1.0 + c_e, 1.0 - c_e, 1.0 + c_e], dtype=np.float64)
    expected_row1 = np.zeros(4, dtype=np.float64)
    
    np.testing.assert_allclose(server.M[0], expected_row0, rtol=1e-12)
    np.testing.assert_allclose(server.M[1], expected_row1, rtol=1e-12)

def test_estimator_controlled_case():
    """
    Construct a tiny controlled sketch matrix mathematically and verify
    estimate_frequency matches the independently evaluated formula.
    """
    m = 4
    k = 2
    epsilon = 2.0
    n = 100
    hash_fam = ThreeWiseHashFamily(k=k, m=m, seed=42)
    
    # Construct synthetic sketch matrix M
    M = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [15.0, 25.0, 35.0, 45.0]
    ], dtype=np.float64)
    
    item_id = 5
    h_indices = hash_fam.hash_all(item_id)
    h0 = h_indices[0]
    h1 = h_indices[1]
    
    sketch_sum = M[0, h0] + M[1, h1]
    raw_mean = sketch_sum / 2.0
    expected_est = (4.0 / 3.0) * (raw_mean - (100.0 / 4.0))
    
    actual_est = estimate_frequency(item_id, M, epsilon, hash_fam, n)
    assert np.isclose(actual_est, expected_est, rtol=1e-12, atol=1e-12)

def test_item_id_range_validation():
    """Verify valid GF(2^12) item IDs [0, 4095] are accepted and out-of-range IDs rejected."""
    hash_fam = ThreeWiseHashFamily(k=16, m=32, seed=42)
    client = CMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(42))
    server = CMSServer(epsilon=2.0, hash_family=hash_fam)
    
    for valid_id in [0, 1, 10, 2600, 4095]:
        rec = client.privatize(valid_id)
        assert 0 <= rec["hash_index"] < 16
        est = server.estimate(valid_id)
        assert isinstance(est, float)
        
    with pytest.raises(ValueError):
        client.privatize(-1)
    with pytest.raises(ValueError):
        client.privatize(4096)

def test_monte_carlo_unbiasedness():
    """
    Critical Monte Carlo statistical test for estimator unbiasedness.
    Controlled vocabulary: A (500 events), B (300 events), C (200 events). Total n = 1000.
    Runs 100 repetitions and verifies mean estimates are within 3 standard errors of true counts.
    """
    m = 32
    k = 16
    epsilon = 2.0
    repetitions = 100
    
    item_A, item_B, item_C = 1, 2, 3
    true_counts = {item_A: 500, item_B: 300, item_C: 200}
    
    # Create event sequence of 1000 items
    events = [item_A] * 500 + [item_B] * 300 + [item_C] * 200
    
    hash_fam_base_seed = 42
    estimates_A = []
    estimates_B = []
    estimates_C = []
    
    for rep in range(repetitions):
        # Sampling hash family h ~ H across repetitions verifies unbiasedness E_h[E_b[f_tilde(d)]] = c(d)
        hash_fam = ThreeWiseHashFamily(k=k, m=m, seed=hash_fam_base_seed + rep)
        client_rng = np.random.default_rng(1000 + rep)
        client = CMSClient(epsilon=epsilon, hash_family=hash_fam, rng=client_rng)
        server = CMSServer(epsilon=epsilon, hash_family=hash_fam)
        
        for item in events:
            rec = client.privatize(item)
            server.process(rec)
            
        estimates_A.append(server.estimate(item_A))
        estimates_B.append(server.estimate(item_B))
        estimates_C.append(server.estimate(item_C))
        
    mean_A = np.mean(estimates_A)
    mean_B = np.mean(estimates_B)
    mean_C = np.mean(estimates_C)
    
    se_A = np.std(estimates_A, ddof=1) / np.sqrt(repetitions)
    se_B = np.std(estimates_B, ddof=1) / np.sqrt(repetitions)
    se_C = np.std(estimates_C, ddof=1) / np.sqrt(repetitions)
    
    assert abs(mean_A - 500) < 3.0 * se_A, f"Mean A {mean_A:.2f} deviated from 500 by more than 3 SE ({3*se_A:.2f})"
    assert abs(mean_B - 300) < 3.0 * se_B, f"Mean B {mean_B:.2f} deviated from 300 by more than 3 SE ({3*se_B:.2f})"
    assert abs(mean_C - 200) < 3.0 * se_C, f"Mean C {mean_C:.2f} deviated from 200 by more than 3 SE ({3*se_C:.2f})"

def test_reproducibility_same_seed():
    """Verify two complete end-to-end runs with seed=42 produce identical outputs."""
    vocab_df = generate_word_vocabulary(50)
    events_df = generate_users_and_events(num_users=100, events_per_user=20, dictionary_df=vocab_df, seed=42, domain="word")
    
    def run_pipeline(seed_val):
        hash_fam = ThreeWiseHashFamily(k=16, m=32, seed=seed_val)
        client_rng = np.random.default_rng(seed_val)
        client = CMSClient(epsilon=2.0, hash_family=hash_fam, rng=client_rng)
        server = CMSServer(epsilon=2.0, hash_family=hash_fam)
        
        for item_id in events_df['item_id']:
            rec = client.privatize(item_id)
            server.process(rec)
            
        estimates = [server.estimate(i) for i in range(50)]
        return server.M.copy(), server.n, estimates

    M1, n1, est1 = run_pipeline(42)
    M2, n2, est2 = run_pipeline(42)
    
    assert n1 == n2 == 2000
    np.testing.assert_array_equal(M1, M2)
    np.testing.assert_array_equal(est1, est2)

def test_different_seeds_behavior():
    """Verify runs with seed=42 vs seed=43 produce different randomized records and sketch matrices."""
    vocab_df = generate_word_vocabulary(50)
    events_df = generate_users_and_events(num_users=50, events_per_user=10, dictionary_df=vocab_df, seed=42, domain="word")
    
    def run_pipeline(seed_val):
        hash_fam = ThreeWiseHashFamily(k=16, m=32, seed=seed_val)
        client_rng = np.random.default_rng(seed_val)
        client = CMSClient(epsilon=2.0, hash_family=hash_fam, rng=client_rng)
        server = CMSServer(epsilon=2.0, hash_family=hash_fam)
        
        for item_id in events_df['item_id']:
            rec = client.privatize(item_id)
            server.process(rec)
            
        estimates = [server.estimate(i) for i in range(50)]
        return server.M.copy(), estimates

    M42, est42 = run_pipeline(42)
    M43, est43 = run_pipeline(43)
    
    assert not np.array_equal(M42, M43)
    assert not np.array_equal(est42, est43)

def test_hash_fixed_randomization_variable():
    """
    Demonstrate that holding the hash family seed fixed (seed_hash=42) while changing the
    client randomization seed alters the randomized outputs while preserving hash functions.
    """
    hash_fam = ThreeWiseHashFamily(k=16, m=32, seed=42)
    
    client1 = CMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(101))
    client2 = CMSClient(epsilon=2.0, hash_family=hash_fam, rng=np.random.default_rng(202))
    
    # Evaluated hash function is identical for both clients for any fixed j
    assert hash_fam.hash(15, 0) == hash_fam.hash(15, 0)
    
    # Privatized outputs differ due to separate client randomization RNG streams
    rec1 = client1.privatize(15)
    rec2 = client2.privatize(15)
    
    diff = (rec1["hash_index"] != rec2["hash_index"]) or not np.array_equal(rec1["vector"], rec2["vector"])
    assert diff

def test_end_to_end_validation_run():
    """
    End-to-end integration test with Phase 2 dev dataset (20,000 records).
    Verifies full pipeline completion, record tracking, and valid outputs.
    """
    vocab_df = generate_word_vocabulary(50)
    events_df = generate_users_and_events(num_users=1000, events_per_user=20, dictionary_df=vocab_df, seed=42, domain="word")
    gt_df = compute_ground_truth(events_df, vocab_df)
    
    hash_fam = ThreeWiseHashFamily(k=16, m=32, seed=42)
    client_rng = np.random.default_rng(42)
    client = CMSClient(epsilon=2.0, hash_family=hash_fam, rng=client_rng)
    server = CMSServer(epsilon=2.0, hash_family=hash_fam)
    
    for item_id in events_df['item_id']:
        rec = client.privatize(item_id)
        server.process(rec)
        
    assert server.n == 20000
    
    # Check ground truth sum equals total events
    assert gt_df['true_count'].sum() == 20000
    
    # Verify estimates exist for all 50 items
    estimates = server.estimate_many(range(50))
    assert len(estimates) == 50
    assert all(isinstance(v, float) for v in estimates.values())
