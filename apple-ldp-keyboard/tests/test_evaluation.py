import os
import pytest
import numpy as np
import pandas as pd
from src.evaluation.metrics import (
    mean_absolute_error, root_mean_square_error, normalized_mae,
    top_k_overlap, spearman_rank_correlation, cms_communication_bytes,
    cms_sketch_memory_bytes
)
from src.evaluation.tables import save_detailed_results_csv, save_summary_results_csv
from src.evaluation.plots import plot_metric_vs_epsilon

def test_mae_known_example():
    true_counts = np.array([100, 50, 25, 10])
    est_counts = np.array([110, 40, 25, 5])  # diffs: 10, 10, 0, 5 -> mean 25/4 = 6.25
    mae = mean_absolute_error(true_counts, est_counts)
    assert np.isclose(mae, 6.25)

def test_rmse_known_example():
    true_counts = np.array([10, 20])
    est_counts = np.array([13, 16])  # diffs: +3, -4 -> squares: 9, 16 -> mean: 12.5 -> sqrt: 3.5355339
    rmse = root_mean_square_error(true_counts, est_counts)
    assert np.isclose(rmse, np.sqrt(12.5))

def test_nmae():
    true_counts = np.array([100, 50])
    est_counts = np.array([110, 40])  # MAE = 10
    nmae = normalized_mae(true_counts, est_counts, total_events=1000)
    assert np.isclose(nmae, 0.01)

def test_top_k_overlap():
    true_counts = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
    # Reverse top 2
    est_counts = np.array([90, 100, 80, 70, 60, 50, 40, 30, 20, 10])
    overlap = top_k_overlap(true_counts, est_counts, k=5)
    assert np.isclose(overlap, 1.0)
    
    # Completely distinct top 2
    est_counts2 = np.array([10, 20, 80, 70, 60, 50, 40, 30, 90, 100])
    overlap2 = top_k_overlap(true_counts, est_counts2, k=2)
    assert np.isclose(overlap2, 0.0)

def test_spearman_rank_correlation():
    true_counts = np.array([10, 20, 30, 40, 50])
    est_counts = np.array([12, 18, 33, 39, 52])
    corr = spearman_rank_correlation(true_counts, est_counts)
    assert np.isclose(corr, 1.0)
    
    # Degenerate constant case returns NaN
    constant_counts = np.array([5, 5, 5, 5])
    assert np.isnan(spearman_rank_correlation(true_counts[:4], constant_counts))

def test_cms_communication_bytes():
    # m=128 -> ceil(128/8) = 16 bytes. k=128 -> ceil(log2(128)/8) = ceil(7/8) = 1 byte -> total = 17 bytes
    assert cms_communication_bytes(m=128, k=128) == 17
    # m=32 -> ceil(32/8) = 4 bytes. k=16 -> ceil(log2(16)/8) = ceil(4/8) = 1 byte -> total = 5 bytes
    assert cms_communication_bytes(m=32, k=16) == 5

def test_cms_sketch_memory_bytes():
    bytes_val, mib_val = cms_sketch_memory_bytes(m=1024, k=65536)
    expected_bytes = 65536 * 1024 * 8  # 536,870,912 bytes
    assert bytes_val == expected_bytes
    assert np.isclose(mib_val, 512.0)

def test_tables_saving(tmp_path):
    detailed_df = pd.DataFrame([{
        "experiment": "test",
        "epsilon": 2.0,
        "users": 100,
        "events_per_user": 20,
        "total_events": 2000,
        "dictionary_size": 50,
        "m": 32,
        "k": 16,
        "repetition": 0,
        "mae": 1.0,
        "rmse": 2.0,
        "nmae": 0.001,
        "top10_overlap": 0.8,
        "spearman": 0.9,
        "communication_bytes_per_record": 5,
        "sketch_memory_bytes": 4096,
        "sketch_memory_mib": 0.0039
    }])
    file_path = os.path.join(tmp_path, "test_detailed.csv")
    save_detailed_results_csv(detailed_df, file_path)
    assert os.path.exists(file_path)
    read_df = pd.read_csv(file_path)
    assert len(read_df) == 1
    assert list(read_df.columns)[0] == "experiment"

def test_plots_creation(tmp_path):
    summary_df = pd.DataFrame([{
        "epsilon": 2.0,
        "mae_mean": 10.0,
        "mae_std": 1.0
    }, {
        "epsilon": 4.0,
        "mae_mean": 5.0,
        "mae_std": 0.5
    }])
    output_path = os.path.join(tmp_path, "plot_test.png")
    plot_metric_vs_epsilon(summary_df, "mae", "MAE", "Test Title", output_path)
    assert os.path.exists(output_path)
