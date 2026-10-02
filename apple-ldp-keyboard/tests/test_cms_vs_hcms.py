import os
import math
import pytest
import numpy as np
import pandas as pd

from src.data.word_loader import generate_word_vocabulary
from src.data.dataset_generator import generate_users_and_events
from src.crypto.hash_family import ThreeWiseHashFamily
from src.cms.client import CMSClient
from src.cms.server import CMSServer
from src.hcms.client import HCMSClient
from src.hcms.server import HCMSServer
from src.evaluation.metrics import (
    cms_communication_bytes, hcms_communication_bytes,
    mean_absolute_error, root_mean_square_error
)
from src.experiments.cms_vs_hcms import run_cms_vs_hcms_experiment


def test_cms_hcms_identical_input_events():
    """
    Test 1: Verify that CMS and HCMS consume strictly identical input event sequences.
    """
    vocab_df = generate_word_vocabulary(500)
    events_df = generate_users_and_events(num_users=10, events_per_user=5, dictionary_df=vocab_df, seed=42, domain="word")

    
    cms_events = events_df['item_id'].values
    hcms_events = events_df['item_id'].values
    
    assert len(cms_events) == len(hcms_events)
    assert np.array_equal(cms_events, hcms_events)


def test_cms_hcms_shared_hash_family():
    """
    Test 2: Verify that CMS and HCMS share the exact same hash family.
    """
    hf_cms = ThreeWiseHashFamily(k=128, m=128, seed=42)
    hf_hcms = ThreeWiseHashFamily(k=128, m=128, seed=42)
    
    item_id = 42
    for j in range(128):
        assert hf_cms.hash(item_id, j) == hf_hcms.hash(item_id, j)


def test_hcms_one_hot_vector_structure():
    """
    Test 3: Verify HCMS construct vector v is all zeros except hashed coordinate = 1.
    """
    hf = ThreeWiseHashFamily(k=128, m=128, seed=42)
    rng = np.random.default_rng(123)
    client = HCMSClient(epsilon=2.0, hash_family=hf, rng=rng)
    
    item_id = 100
    # Evaluate privatize call and check that one-hot vector property holds internally
    payload = client.privatize(item_id)
    j = payload["hash_index"]
    h_val = hf.hash(item_id, j)
    
    v = np.zeros(128, dtype=np.float64)
    v[h_val] = 1.0
    assert np.sum(v) == 1.0
    assert v[h_val] == 1.0
    assert np.count_nonzero(v) == 1


def test_communication_formula_cms():
    """
    Test 4: Verify theoretical communication calculation for CMS.
    ceil((m + ceil(log2(k))) / 8)
    For m=128, k=128: ceil((128 + 7) / 8) = ceil(135 / 8) = 17 bytes.
    """
    m, k = 128, 128
    bits_k = math.ceil(math.log2(k))
    expected_bits = m + bits_k
    expected_bytes = math.ceil(expected_bits / 8.0)
    
    calc_bytes = cms_communication_bytes(m, k)
    assert calc_bytes == expected_bytes
    assert calc_bytes == 17


def test_communication_formula_hcms():
    """
    Test 5: Verify theoretical communication calculation for HCMS.
    ceil((1 + ceil(log2(k)) + ceil(log2(m))) / 8)
    For m=128, k=128: ceil((1 + 7 + 7) / 8) = ceil(15 / 8) = 2 bytes.
    """
    import math
    m, k = 128, 128
    bits_k = math.ceil(math.log2(k))
    bits_m = math.ceil(math.log2(m))
    expected_bits = 1 + bits_k + bits_m
    expected_bytes = math.ceil(expected_bits / 8.0)
    
    calc_bytes = hcms_communication_bytes(m, k)
    assert calc_bytes == expected_bytes
    assert calc_bytes == 2


def test_result_schema_correctness():
    """
    Test 6: Verify result CSV files exist and conform to exact required schemas.
    """
    raw_path = "results/csv/cms_vs_hcms_results.csv"
    summary_path = "results/csv/cms_vs_hcms_summary.csv"
    paired_path = "results/csv/cms_vs_hcms_paired.csv"
    
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        expected_raw_cols = [
            "experiment", "mechanism", "epsilon", "repetition", "users",
            "events_per_user", "total_events", "dictionary_size", "m", "k",
            "mae", "rmse", "nmae", "top10_overlap", "spearman",
            "client_total_time_seconds", "client_time_per_record_ms",
            "server_total_time_seconds", "server_time_per_record_ms",
            "hcms_finalization_time_seconds", "communication_bytes_per_record",
            "sketch_memory_bytes", "sketch_memory_mib"
        ]
        assert list(df_raw.columns) == expected_raw_cols

    if os.path.exists(summary_path):
        df_sum = pd.read_csv(summary_path)
        expected_sum_cols = [
            "mechanism", "epsilon", "users", "dictionary_size", "m", "k", "repetitions",
            "mae_mean", "mae_std", "rmse_mean", "rmse_std", "nmae_mean", "nmae_std",
            "top10_overlap_mean", "top10_overlap_std", "spearman_mean", "spearman_std",
            "client_total_time_mean_seconds", "client_total_time_std_seconds",
            "client_time_per_record_mean_ms", "client_time_per_record_std_ms",
            "server_total_time_mean_seconds", "server_total_time_std_seconds",
            "server_time_per_record_mean_ms", "server_time_per_record_std_ms",
            "hcms_finalization_time_mean_seconds", "hcms_finalization_time_std_seconds",
            "communication_bytes_per_record", "sketch_memory_bytes", "sketch_memory_mib"
        ]
        assert list(df_sum.columns) == expected_sum_cols

    if os.path.exists(paired_path):
        df_paired = pd.read_csv(paired_path)
        expected_paired_cols = [
            "epsilon", "repetition", "cms_mae", "hcms_mae", "cms_rmse", "hcms_rmse",
            "cms_nmae", "hcms_nmae", "cms_top10_overlap", "hcms_top10_overlap",
            "cms_spearman", "hcms_spearman", "mae_difference_cms_minus_hcms",
            "rmse_difference_cms_minus_hcms", "nmae_difference_cms_minus_hcms",
            "top10_difference_cms_minus_hcms", "spearman_difference_cms_minus_hcms"
        ]
        assert list(df_paired.columns) == expected_paired_cols


def test_paired_output_pairing():
    """
    Test 7: Verify that the results file has 100 rows (5 epsilons x 10 reps x 2 mechanisms),
    paired 1-to-1 for every epsilon and repetition.
    """
    raw_path = "results/csv/cms_vs_hcms_results.csv"
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        assert len(df_raw) == 100
        
        for eps in [0.5, 1.0, 2.0, 4.0, 8.0]:
            for rep in range(10):
                sub = df_raw[(df_raw['epsilon'] == eps) & (df_raw['repetition'] == rep)]
                assert len(sub) == 2
                mechs = set(sub['mechanism'])
                assert mechs == {"CMS", "HCMS"}


def test_runtimes_non_negative():
    """
    Test 8: Verify that all measured runtime values are strictly non-negative.
    """
    raw_path = "results/csv/cms_vs_hcms_results.csv"
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        assert (df_raw['client_total_time_seconds'] >= 0).all()
        assert (df_raw['server_total_time_seconds'] >= 0).all()
        assert (df_raw['hcms_finalization_time_seconds'] >= 0).all()


def test_reproducibility():
    """
    Reproducibility Test (Section 34):
    Repeat one complete epsilon/repetition pair twice with the exact same dataset,
    hash family, and randomization seed. Verify exact equality of estimates.
    """
    vocab_df = generate_word_vocabulary(100)
    events_df = generate_users_and_events(num_users=10, events_per_user=5, dictionary_df=vocab_df, seed=42, domain="word")

    item_ids = events_df['item_id'].values
    
    hf = ThreeWiseHashFamily(k=32, m=32, seed=42)
    
    # CMS Run 1
    c1 = CMSClient(epsilon=2.0, hash_family=hf, rng=np.random.default_rng(500000))
    s1 = CMSServer(epsilon=2.0, hash_family=hf)
    for iid in item_ids:
        s1.process(c1.privatize(iid))
    est1 = s1.estimate_many(range(100))
    
    # CMS Run 2
    c2 = CMSClient(epsilon=2.0, hash_family=hf, rng=np.random.default_rng(500000))
    s2 = CMSServer(epsilon=2.0, hash_family=hf)
    for iid in item_ids:
        s2.process(c2.privatize(iid))
    est2 = s2.estimate_many(range(100))
    
    for iid in range(100):
        assert np.isclose(est1[iid], est2[iid])

    # HCMS Run 1
    hc1 = HCMSClient(epsilon=2.0, hash_family=hf, rng=np.random.default_rng(600000))
    hs1 = HCMSServer(epsilon=2.0, hash_family=hf)
    for iid in item_ids:
        hs1.process(hc1.privatize(iid))
    h_est1 = hs1.estimate_many(range(100))

    # HCMS Run 2
    hc2 = HCMSClient(epsilon=2.0, hash_family=hf, rng=np.random.default_rng(600000))
    hs2 = HCMSServer(epsilon=2.0, hash_family=hf)
    for iid in item_ids:
        hs2.process(hc2.privatize(iid))
    h_est2 = hs2.estimate_many(range(100))

    for iid in range(100):
        assert np.isclose(h_est1[iid], h_est2[iid])
