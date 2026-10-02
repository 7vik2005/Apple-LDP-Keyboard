import os
import time
import logging
import pandas as pd
import numpy as np

from src.data.word_loader import generate_word_vocabulary
from src.data.dataset_generator import generate_users_and_events, compute_ground_truth
from src.crypto.hash_family import ThreeWiseHashFamily
from src.cms.client import CMSClient
from src.cms.server import CMSServer
from src.hcms.client import HCMSClient
from src.hcms.server import HCMSServer
from src.evaluation.metrics import (
    mean_absolute_error, root_mean_square_error, normalized_mae,
    top_k_overlap, spearman_rank_correlation,
    cms_communication_bytes, hcms_communication_bytes,
    cms_sketch_memory_bytes
)
from src.evaluation.plots import plot_comparison_vs_epsilon, plot_categorical_comparison


def setup_logger():
    os.makedirs("results/logs", exist_ok=True)
    logger = logging.getLogger("phase8_cms_vs_hcms")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("results/logs/phase8.log", mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def validate_result_sanity(res: dict) -> None:
    """
    Validate that metric and timing values meet strict physical and mathematical bounds.
    Raises ValueError if any metric is invalid.
    """
    if res["mae"] < 0:
        raise ValueError(f"Invalid MAE: {res['mae']}")
    if res["rmse"] < 0:
        raise ValueError(f"Invalid RMSE: {res['rmse']}")
    if res["nmae"] < 0:
        raise ValueError(f"Invalid NMAE: {res['nmae']}")
    if not (0.0 <= res["top10_overlap"] <= 1.0):
        raise ValueError(f"Invalid Top-10 Overlap: {res['top10_overlap']}")
    if not np.isnan(res["spearman"]) and not (-1.0 <= res["spearman"] <= 1.0):
        raise ValueError(f"Invalid Spearman: {res['spearman']}")
    if res["client_total_time_seconds"] < 0:
        raise ValueError(f"Invalid client runtime: {res['client_total_time_seconds']}")
    if res["server_total_time_seconds"] < 0:
        raise ValueError(f"Invalid server runtime: {res['server_total_time_seconds']}")
    if res["hcms_finalization_time_seconds"] < 0:
        raise ValueError(f"Invalid finalization runtime: {res['hcms_finalization_time_seconds']}")
    if res["communication_bytes_per_record"] <= 0:
        raise ValueError(f"Invalid communication bytes: {res['communication_bytes_per_record']}")
    if res["sketch_memory_bytes"] <= 0:
        raise ValueError(f"Invalid sketch memory bytes: {res['sketch_memory_bytes']}")


def run_cms_vs_hcms_experiment():
    logger = setup_logger()
    logger.info("Starting Phase 8 Experiment: CMS vs HCMS Comparative Evaluation")

    # Fixed Parameters
    dictionary_size = 500
    num_users = 1000
    events_per_user = 20
    total_events = num_users * events_per_user
    m = 128
    k = 128
    epsilon_values = [0.5, 1.0, 2.0, 4.0, 8.0]
    repetitions = 10
    dataset_seed = 42
    hash_seed = 42

    # Step 1: Load fixed 500-word dataset & ground truth
    vocab_df = generate_word_vocabulary(dictionary_size)
    events_df = generate_users_and_events(
        num_users=num_users,
        events_per_user=events_per_user,
        dictionary_df=vocab_df,
        seed=dataset_seed,
        domain="word"
    )
    gt_df = compute_ground_truth(events_df, vocab_df)

    assert gt_df['true_count'].sum() == total_events
    assert len(gt_df) == dictionary_size

    true_counts = gt_df['true_count'].values
    item_ids = gt_df['item_id'].values
    input_event_ids = events_df['item_id'].values

    # Step 2: Construct ONE deterministic hash family
    hash_family = ThreeWiseHashFamily(k=k, m=m, seed=hash_seed)

    # Theoretical communication and sketch memory calculations
    cms_comm_bytes = cms_communication_bytes(m, k)
    hcms_comm_bytes = hcms_communication_bytes(m, k)
    mem_bytes, mem_mib = cms_sketch_memory_bytes(m, k)

    logger.info(f"Dataset loaded: {total_events} events, dict size {dictionary_size}")
    logger.info(f"CMS payload: {cms_comm_bytes} bytes/record | HCMS payload: {hcms_comm_bytes} bytes/record")
    logger.info(f"Sketch memory: {mem_bytes} bytes ({mem_mib:.4f} MiB)")

    raw_results = []
    paired_rows = []

    for e_idx, eps in enumerate(epsilon_values):
        logger.info(f"Running comparison for epsilon={eps}")

        for r in range(repetitions):
            # Deterministic randomization seeds per section 9
            cms_seed = 500000 + e_idx * 100 + r
            hcms_seed = 600000 + e_idx * 100 + r

            # --- COMMON INPUT VALIDATION (Section 35) ---
            # Input event sequences for CMS and HCMS must be strictly identical
            cms_events = input_event_ids
            hcms_events = input_event_ids
            assert len(cms_events) == len(hcms_events)
            assert np.array_equal(cms_events, hcms_events)

            # ==========================================
            # 1. RUN CMS
            # ==========================================
            cms_client_rng = np.random.default_rng(cms_seed)
            cms_client = CMSClient(epsilon=eps, hash_family=hash_family, rng=cms_client_rng)
            cms_server = CMSServer(epsilon=eps, hash_family=hash_family)

            # Measure CMS client processing
            t0 = time.perf_counter()
            cms_records = [cms_client.privatize(iid) for iid in cms_events]
            t_client_cms = time.perf_counter() - t0

            # Measure CMS server record ingestion
            t0 = time.perf_counter()
            for record in cms_records:
                cms_server.process(record)
            t_server_cms = time.perf_counter() - t0

            # CMS estimates & metrics
            cms_est_dict = cms_server.estimate_many(item_ids)
            cms_est_counts = np.array([cms_est_dict[iid] for iid in item_ids])

            cms_mae = mean_absolute_error(true_counts, cms_est_counts)
            cms_rmse = root_mean_square_error(true_counts, cms_est_counts)
            cms_nmae = normalized_mae(true_counts, cms_est_counts, total_events)
            cms_top10 = top_k_overlap(true_counts, cms_est_counts, k=10, item_ids=item_ids)
            cms_spearman = spearman_rank_correlation(true_counts, cms_est_counts)

            cms_res = {
                "experiment": "cms_vs_hcms_comparison",
                "mechanism": "CMS",
                "epsilon": float(eps),
                "repetition": r,
                "users": num_users,
                "events_per_user": events_per_user,
                "total_events": total_events,
                "dictionary_size": dictionary_size,
                "m": m,
                "k": k,
                "mae": cms_mae,
                "rmse": cms_rmse,
                "nmae": cms_nmae,
                "top10_overlap": cms_top10,
                "spearman": cms_spearman,
                "client_total_time_seconds": t_client_cms,
                "client_time_per_record_ms": (t_client_cms / total_events) * 1000.0,
                "server_total_time_seconds": t_server_cms,
                "server_time_per_record_ms": (t_server_cms / total_events) * 1000.0,
                "hcms_finalization_time_seconds": 0.0,
                "communication_bytes_per_record": cms_comm_bytes,
                "sketch_memory_bytes": mem_bytes,
                "sketch_memory_mib": mem_mib
            }
            validate_result_sanity(cms_res)
            raw_results.append(cms_res)

            # ==========================================
            # 2. RUN HCMS
            # ==========================================
            hcms_client_rng = np.random.default_rng(hcms_seed)
            hcms_client = HCMSClient(epsilon=eps, hash_family=hash_family, rng=hcms_client_rng)
            hcms_server = HCMSServer(epsilon=eps, hash_family=hash_family)

            # Measure HCMS client processing
            t0 = time.perf_counter()
            hcms_records = [hcms_client.privatize(iid) for iid in hcms_events]
            t_client_hcms = time.perf_counter() - t0

            # Measure HCMS server record ingestion (sparse sketch update)
            t0 = time.perf_counter()
            for record in hcms_records:
                hcms_server.process(record)
            t_server_rec_hcms = time.perf_counter() - t0

            # Measure HCMS server finalization (Hadamard reconstruction)
            t0 = time.perf_counter()
            hcms_server.finalize()
            t_finalization_hcms = time.perf_counter() - t0

            t_server_hcms_total = t_server_rec_hcms + t_finalization_hcms

            # HCMS estimates & metrics
            hcms_est_dict = hcms_server.estimate_many(item_ids)
            hcms_est_counts = np.array([hcms_est_dict[iid] for iid in item_ids])

            hcms_mae = mean_absolute_error(true_counts, hcms_est_counts)
            hcms_rmse = root_mean_square_error(true_counts, hcms_est_counts)
            hcms_nmae = normalized_mae(true_counts, hcms_est_counts, total_events)
            hcms_top10 = top_k_overlap(true_counts, hcms_est_counts, k=10, item_ids=item_ids)
            hcms_spearman = spearman_rank_correlation(true_counts, hcms_est_counts)

            hcms_res = {
                "experiment": "cms_vs_hcms_comparison",
                "mechanism": "HCMS",
                "epsilon": float(eps),
                "repetition": r,
                "users": num_users,
                "events_per_user": events_per_user,
                "total_events": total_events,
                "dictionary_size": dictionary_size,
                "m": m,
                "k": k,
                "mae": hcms_mae,
                "rmse": hcms_rmse,
                "nmae": hcms_nmae,
                "top10_overlap": hcms_top10,
                "spearman": hcms_spearman,
                "client_total_time_seconds": t_client_hcms,
                "client_time_per_record_ms": (t_client_hcms / total_events) * 1000.0,
                "server_total_time_seconds": t_server_hcms_total,
                "server_time_per_record_ms": (t_server_hcms_total / total_events) * 1000.0,
                "hcms_finalization_time_seconds": t_finalization_hcms,
                "communication_bytes_per_record": hcms_comm_bytes,
                "sketch_memory_bytes": mem_bytes,
                "sketch_memory_mib": mem_mib
            }
            validate_result_sanity(hcms_res)
            raw_results.append(hcms_res)

            # ==========================================
            # 3. PAIRED DIFFERENCES (Section 18 & 19)
            # ==========================================
            paired_rows.append({
                "epsilon": float(eps),
                "repetition": r,
                "cms_mae": cms_mae,
                "hcms_mae": hcms_mae,
                "cms_rmse": cms_rmse,
                "hcms_rmse": hcms_rmse,
                "cms_nmae": cms_nmae,
                "hcms_nmae": hcms_nmae,
                "cms_top10_overlap": cms_top10,
                "hcms_top10_overlap": hcms_top10,
                "cms_spearman": cms_spearman,
                "hcms_spearman": hcms_spearman,
                "mae_difference_cms_minus_hcms": cms_mae - hcms_mae,
                "rmse_difference_cms_minus_hcms": cms_rmse - hcms_rmse,
                "nmae_difference_cms_minus_hcms": cms_nmae - hcms_nmae,
                "top10_difference_cms_minus_hcms": cms_top10 - hcms_top10,
                "spearman_difference_cms_minus_hcms": cms_spearman - hcms_spearman
            })

    # Save CSV files per schema
    os.makedirs("results/csv", exist_ok=True)
    raw_df = pd.DataFrame(raw_results)
    raw_csv_path = "results/csv/cms_vs_hcms_results.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    logger.info(f"Saved raw detailed comparison CSV: {raw_csv_path}")

    paired_df = pd.DataFrame(paired_rows)
    paired_csv_path = "results/csv/cms_vs_hcms_paired.csv"
    paired_df.to_csv(paired_csv_path, index=False)
    logger.info(f"Saved paired differences CSV: {paired_csv_path}")

    # Generate Summary CSV per section 17
    summary_rows = []
    grouped = raw_df.groupby(["mechanism", "epsilon"], as_index=False)
    for (mech, eps), group in grouped:
        summary_rows.append({
            "mechanism": mech,
            "epsilon": float(eps),
            "users": num_users,
            "dictionary_size": dictionary_size,
            "m": m,
            "k": k,
            "repetitions": repetitions,
            "mae_mean": group["mae"].mean(),
            "mae_std": group["mae"].std(),
            "rmse_mean": group["rmse"].mean(),
            "rmse_std": group["rmse"].std(),
            "nmae_mean": group["nmae"].mean(),
            "nmae_std": group["nmae"].std(),
            "top10_overlap_mean": group["top10_overlap"].mean(),
            "top10_overlap_std": group["top10_overlap"].std(),
            "spearman_mean": group["spearman"].mean(),
            "spearman_std": group["spearman"].std(),
            "client_total_time_mean_seconds": group["client_total_time_seconds"].mean(),
            "client_total_time_std_seconds": group["client_total_time_seconds"].std(),
            "client_time_per_record_mean_ms": group["client_time_per_record_ms"].mean(),
            "client_time_per_record_std_ms": group["client_time_per_record_ms"].std(),
            "server_total_time_mean_seconds": group["server_total_time_seconds"].mean(),
            "server_total_time_std_seconds": group["server_total_time_seconds"].std(),
            "server_time_per_record_mean_ms": group["server_time_per_record_ms"].mean(),
            "server_time_per_record_std_ms": group["server_time_per_record_ms"].std(),
            "hcms_finalization_time_mean_seconds": group["hcms_finalization_time_seconds"].mean(),
            "hcms_finalization_time_std_seconds": group["hcms_finalization_time_seconds"].std(),
            "communication_bytes_per_record": group["communication_bytes_per_record"].iloc[0],
            "sketch_memory_bytes": group["sketch_memory_bytes"].iloc[0],
            "sketch_memory_mib": group["sketch_memory_mib"].iloc[0]
        })

    summary_df = pd.DataFrame(summary_rows)
    # Reorder columns to strictly match section 17
    summary_cols = [
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
    summary_df = summary_df[summary_cols]
    summary_csv_path = "results/csv/cms_vs_hcms_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved summary comparison CSV: {summary_csv_path}")

    # Generate Plots (Sections 21 - 29)
    os.makedirs("results/plots", exist_ok=True)
    plot_comparison_vs_epsilon(summary_df, "mae", "Mean Absolute Error (MAE)", "MAE vs Epsilon: CMS vs HCMS", "results/plots/cms_vs_hcms_mae.png")
    plot_comparison_vs_epsilon(summary_df, "rmse", "Root Mean Square Error (RMSE)", "RMSE vs Epsilon: CMS vs HCMS", "results/plots/cms_vs_hcms_rmse.png")
    plot_comparison_vs_epsilon(summary_df, "nmae", "Normalized MAE (NMAE)", "NMAE vs Epsilon: CMS vs HCMS", "results/plots/cms_vs_hcms_nmae.png")
    plot_comparison_vs_epsilon(summary_df, "top10_overlap", "Top-10 Overlap", "Top-10 Overlap vs Epsilon: CMS vs HCMS", "results/plots/cms_vs_hcms_top10.png", ylim=(0.0, 1.0))
    plot_comparison_vs_epsilon(summary_df, "spearman", "Spearman Rank Correlation", "Spearman Correlation vs Epsilon: CMS vs HCMS", "results/plots/cms_vs_hcms_spearman.png")
    plot_comparison_vs_epsilon(summary_df, "client_total_time", "Client Runtime (seconds)", "Client Runtime vs Epsilon: CMS vs HCMS", "results/plots/cms_vs_hcms_client_runtime.png")
    plot_comparison_vs_epsilon(summary_df, "server_total_time", "Server Runtime (seconds)", "Server Runtime vs Epsilon: CMS vs HCMS", "results/plots/cms_vs_hcms_server_runtime.png")

    plot_categorical_comparison(["CMS", "HCMS"], [cms_comm_bytes, hcms_comm_bytes], "Bytes per Record", "Client Communication Payload: CMS vs HCMS", "results/plots/cms_vs_hcms_communication.png")
    plot_categorical_comparison(["CMS", "HCMS"], [mem_mib, mem_mib], "Sketch Memory (MiB)", "Server Sketch Memory: CMS vs HCMS", "results/plots/cms_vs_hcms_memory.png")

    logger.info("Phase 8 experiment execution completed successfully.")
    print("=== Phase 8 CMS vs HCMS Comparative Evaluation Complete ===")
    print(f"Summary saved to: {summary_csv_path}")
    print(f"Plots saved to: results/plots/")


if __name__ == "__main__":
    run_cms_vs_hcms_experiment()
