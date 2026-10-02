import os
import json
import logging
import pandas as pd
import numpy as np

from src.data.emoji_loader import load_emoji_vocabulary
from src.data.dataset_generator import generate_users_and_events, compute_ground_truth
from src.crypto.hash_family import ThreeWiseHashFamily
from src.cms.client import CMSClient
from src.cms.server import CMSServer
from src.evaluation.metrics import (
    mean_absolute_error, root_mean_square_error, normalized_mae,
    top_k_overlap, spearman_rank_correlation,
    cms_communication_bytes, cms_sketch_memory_bytes
)
from src.evaluation.plots import plot_metric_vs_epsilon


def setup_logger():
    os.makedirs("results/logs", exist_ok=True)
    logger = logging.getLogger("phase9_apple_replication")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("results/logs/phase9.log", mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def run_apple_replication_experiment():
    logger = setup_logger()
    logger.info("Starting Phase 9 Experiment: Apple-Parameter Experiment")

    # Load configuration from configs/apple_like.json
    config_path = "configs/apple_like.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    dictionary_size = config["dictionary_size"]
    epsilon = config["epsilon"]
    m = config["cms"]["m"]
    k = config["cms"]["k"]
    repetitions = config["repetitions"]
    dataset_seed = config["seed"]
    hash_seed = config["seed"]

    # Fixed Population
    num_users = 1000
    events_per_user = 20
    total_events = num_users * events_per_user

    # Load/generate 2600-emoji vocabulary
    os.makedirs("data/generated", exist_ok=True)
    emoji_df = load_emoji_vocabulary(dictionary_size=dictionary_size, s=1.07)
    emoji_vocab_path = "data/generated/emoji_vocabulary.csv"
    emoji_df.to_csv(emoji_vocab_path, index=False)
    logger.info(f"Saved emoji vocabulary to {emoji_vocab_path}")

    # Generate fixed events
    events_df = generate_users_and_events(
        num_users=num_users,
        events_per_user=events_per_user,
        dictionary_df=emoji_df,
        seed=dataset_seed,
        domain="emoji"
    )
    gt_df = compute_ground_truth(events_df, emoji_df)

    assert len(gt_df) == dictionary_size
    assert gt_df['true_count'].sum() == total_events

    true_counts = gt_df['true_count'].values
    item_ids = gt_df['item_id'].values
    input_events = events_df['item_id'].values

    # Construct Phase-3 Hash Family (k=65536, m=1024, seed=42)
    logger.info(f"Constructing ThreeWiseHashFamily k={k}, m={m}, seed={hash_seed}...")
    hash_family = ThreeWiseHashFamily(k=k, m=m, seed=hash_seed)

    comm_bytes = cms_communication_bytes(m, k)
    mem_bytes, mem_mib = cms_sketch_memory_bytes(m, k)

    logger.info(f"Theoretical CMS payload: {comm_bytes} bytes/record")
    logger.info(f"Theoretical CMS memory: {mem_bytes} bytes ({mem_mib:.2f} MiB)")

    detailed_rows = []

    for r in range(repetitions):
        rand_seed = 900000 + r
        logger.info(f"Running Apple-replication repetition r={r} with seed={rand_seed}")
        client_rng = np.random.default_rng(rand_seed)

        client = CMSClient(epsilon=epsilon, hash_family=hash_family, rng=client_rng)
        server = CMSServer(epsilon=epsilon, hash_family=hash_family)

        for item_id in input_events:
            record = client.privatize(item_id)
            server.process(record)

        estimates_dict = server.estimate_many(item_ids)
        est_counts = np.array([estimates_dict[iid] for iid in item_ids])

        mae = mean_absolute_error(true_counts, est_counts)
        rmse = root_mean_square_error(true_counts, est_counts)
        nmae = normalized_mae(true_counts, est_counts, total_events)
        top10 = top_k_overlap(true_counts, est_counts, k=10, item_ids=item_ids)
        spearman = spearman_rank_correlation(true_counts, est_counts)

        detailed_rows.append({
            "experiment": "apple_replication",
            "repetition": r,
            "epsilon": float(epsilon),
            "users": num_users,
            "events_per_user": events_per_user,
            "total_events": total_events,
            "dictionary_size": dictionary_size,
            "m": m,
            "k": k,
            "mae": mae,
            "rmse": rmse,
            "nmae": nmae,
            "top10_overlap": top10,
            "spearman": spearman,
            "communication_bytes_per_record": comm_bytes,
            "sketch_memory_bytes": mem_bytes,
            "sketch_memory_mib": mem_mib
        })

    os.makedirs("results/csv", exist_ok=True)
    raw_df = pd.DataFrame(detailed_rows)
    raw_csv_path = "results/csv/apple_replication_results.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    logger.info(f"Saved raw Apple-replication results: {raw_csv_path}")

    summary_row = {
        "epsilon": float(epsilon),
        "users": num_users,
        "events_per_user": events_per_user,
        "total_events": total_events,
        "dictionary_size": dictionary_size,
        "m": m,
        "k": k,
        "repetitions": repetitions,
        "mae_mean": raw_df["mae"].mean(),
        "mae_std": raw_df["mae"].std(),
        "rmse_mean": raw_df["rmse"].mean(),
        "rmse_std": raw_df["rmse"].std(),
        "nmae_mean": raw_df["nmae"].mean(),
        "nmae_std": raw_df["nmae"].std(),
        "top10_overlap_mean": raw_df["top10_overlap"].mean(),
        "top10_overlap_std": raw_df["top10_overlap"].std(),
        "spearman_mean": raw_df["spearman"].mean(),
        "spearman_std": raw_df["spearman"].std(),
        "communication_bytes_per_record": comm_bytes,
        "sketch_memory_bytes": mem_bytes,
        "sketch_memory_mib": mem_mib
    }
    summary_df = pd.DataFrame([summary_row])
    summary_csv_path = "results/csv/apple_replication_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved summary Apple-replication CSV: {summary_csv_path}")

    # Generate Parameter Fidelity Table per Section 9.14
    fidelity_rows = [
        {
            "parameter": "epsilon",
            "apple_published_value": 4.0,
            "our_value": float(epsilon),
            "match": float(epsilon) == 4.0,
            "note": "Matches Apple's published Popular Emojis epsilon setting."
        },
        {
            "parameter": "m",
            "apple_published_value": 1024,
            "our_value": int(m),
            "match": int(m) == 1024,
            "note": "Matches Apple's published sketch width."
        },
        {
            "parameter": "k",
            "apple_published_value": 65536,
            "our_value": int(k),
            "match": int(k) == 65536,
            "note": "Matches Apple's published number of hash functions."
        },
        {
            "parameter": "dictionary_size",
            "apple_published_value": 2600,
            "our_value": int(dictionary_size),
            "match": int(dictionary_size) == 2600,
            "note": "Matches Apple's published emoji dictionary domain size; dataset and population are simulated."
        }
    ]
    fidelity_df = pd.DataFrame(fidelity_rows)
    fidelity_csv_path = "results/csv/apple_parameter_fidelity.csv"
    fidelity_df.to_csv(fidelity_csv_path, index=False)
    logger.info(f"Saved parameter fidelity table: {fidelity_csv_path}")

    # Generate Summary Plots (Single categorical/summary point for fixed setting)
    os.makedirs("results/plots", exist_ok=True)
    
    # Custom single-setting summary plot helper
    import matplotlib.pyplot as plt
    metrics_to_plot = [
        ("mae", "Mean Absolute Error (MAE)", "Apple-Parameter Regime: MAE", "results/plots/apple_replication_mae.png"),
        ("rmse", "Root Mean Square Error (RMSE)", "Apple-Parameter Regime: RMSE", "results/plots/apple_replication_rmse.png"),
        ("nmae", "Normalized MAE (NMAE)", "Apple-Parameter Regime: NMAE", "results/plots/apple_replication_nmae.png"),
        ("top10_overlap", "Top-10 Overlap", "Apple-Parameter Regime: Top-10 Overlap", "results/plots/apple_replication_top10.png"),
        ("spearman", "Spearman Rank Correlation", "Apple-Parameter Regime: Spearman Correlation", "results/plots/apple_replication_spearman.png")
    ]

    for metric_key, ylabel, title, out_path in metrics_to_plot:
        plt.figure()
        mean_val = summary_df[f"{metric_key}_mean"].iloc[0]
        std_val = summary_df[f"{metric_key}_std"].iloc[0]
        plt.errorbar(["Apple Reg (m=1024, k=65536, ε=4)"], [mean_val], yerr=[std_val], fmt='o', capsize=6)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        if metric_key == "top10_overlap":
            plt.ylim((0.0, 1.0))
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    logger.info("Phase 9 experiment completed successfully.")
    print("=== Phase 9 Apple-Parameter Experiment Complete ===")
    print(f"Summary saved to: {summary_csv_path}")
    print(f"Fidelity table saved to: {fidelity_csv_path}")


if __name__ == "__main__":
    run_apple_replication_experiment()
