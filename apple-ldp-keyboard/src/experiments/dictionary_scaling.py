import os
import logging
import pandas as pd
import numpy as np

from src.data.word_loader import generate_word_vocabulary
from src.data.dataset_generator import generate_users_and_events, compute_ground_truth
from src.crypto.hash_family import ThreeWiseHashFamily
from src.cms.client import CMSClient
from src.cms.server import CMSServer
from src.evaluation.metrics import (
    mean_absolute_error, root_mean_square_error, normalized_mae,
    top_k_overlap, spearman_rank_correlation, cms_communication_bytes,
    cms_sketch_memory_bytes
)
from src.evaluation.tables import save_detailed_results_csv, save_summary_results_csv
from src.evaluation.plots import plot_metric_vs_dictionary_size

def setup_logger():
    os.makedirs("results/logs", exist_ok=True)
    logger = logging.getLogger("phase6_dictionary_scaling")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("results/logs/phase6.log", mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def run_dictionary_scaling_experiment():
    logger = setup_logger()
    logger.info("Starting Experiment 3: Dictionary-Size Sensitivity")
    
    # Fixed Parameters
    dictionary_sizes = [50, 100, 250, 500]
    num_users = 1000
    events_per_user = 20
    total_events = num_users * events_per_user
    epsilon = 2.0
    m = 128
    k = 128
    repetitions = 10
    base_seed = 42
    experiment_offset = 3000
    
    # Fixed Hash Family
    hash_family = ThreeWiseHashFamily(k=k, m=m, seed=base_seed)
    
    comm_bytes = cms_communication_bytes(m, k)
    mem_bytes, mem_mib = cms_sketch_memory_bytes(m, k)
    
    detailed_rows = []
    
    for d_size in dictionary_sizes:
        logger.info(f"Running Dictionary-Size Sensitivity for dictionary_size={d_size}")
        
        vocab_df = generate_word_vocabulary(d_size)
        events_df = generate_users_and_events(
            num_users=num_users,
            events_per_user=events_per_user,
            dictionary_df=vocab_df,
            seed=base_seed,
            domain="word"
        )
        gt_df = compute_ground_truth(events_df, vocab_df)
        
        assert gt_df['true_count'].sum() == total_events
        assert len(gt_df) == d_size
        
        true_counts = gt_df['true_count'].values
        item_ids = gt_df['item_id'].values
        
        for r in range(repetitions):
            rand_seed = 100000 + experiment_offset + r
            client_rng = np.random.default_rng(rand_seed)
            
            client = CMSClient(epsilon=epsilon, hash_family=hash_family, rng=client_rng)
            server = CMSServer(epsilon=epsilon, hash_family=hash_family)
            
            for item_id in events_df['item_id']:
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
                "experiment": "dictionary_scaling",
                "epsilon": float(epsilon),
                "users": num_users,
                "events_per_user": events_per_user,
                "total_events": total_events,
                "dictionary_size": d_size,
                "m": m,
                "k": k,
                "repetition": r,
                "mae": mae,
                "rmse": rmse,
                "nmae": nmae,
                "top10_overlap": top10,
                "spearman": spearman,
                "communication_bytes_per_record": comm_bytes,
                "sketch_memory_bytes": mem_bytes,
                "sketch_memory_mib": mem_mib
            })
            
    df_detailed = pd.DataFrame(detailed_rows)
    detailed_csv_path = "results/csv/dictionary_scaling_results.csv"
    save_detailed_results_csv(df_detailed, detailed_csv_path)
    logger.info(f"Saved detailed results to {detailed_csv_path}")
    
    # Compute Summary
    summary_rows = []
    for d_size in dictionary_sizes:
        sub_df = df_detailed[df_detailed['dictionary_size'] == d_size]
        summary_rows.append({
            "dictionary_size": d_size,
            "epsilon": float(epsilon),
            "users": num_users,
            "m": m,
            "k": k,
            "repetitions": repetitions,
            "mae_mean": sub_df['mae'].mean(),
            "mae_std": sub_df['mae'].std(ddof=1),
            "rmse_mean": sub_df['rmse'].mean(),
            "rmse_std": sub_df['rmse'].std(ddof=1),
            "nmae_mean": sub_df['nmae'].mean(),
            "nmae_std": sub_df['nmae'].std(ddof=1),
            "top10_overlap_mean": sub_df['top10_overlap'].mean(),
            "top10_overlap_std": sub_df['top10_overlap'].std(ddof=1),
            "spearman_mean": sub_df['spearman'].mean(),
            "spearman_std": sub_df['spearman'].std(ddof=1),
            "communication_bytes_per_record": comm_bytes,
            "sketch_memory_bytes": mem_bytes,
            "sketch_memory_mib": mem_mib
        })
        
    df_summary = pd.DataFrame(summary_rows)
    summary_cols = [
        "dictionary_size", "epsilon", "users", "m", "k", "repetitions",
        "mae_mean", "mae_std", "rmse_mean", "rmse_std", "nmae_mean", "nmae_std",
        "top10_overlap_mean", "top10_overlap_std", "spearman_mean", "spearman_std",
        "communication_bytes_per_record", "sketch_memory_bytes", "sketch_memory_mib"
    ]
    summary_csv_path = "results/csv/dictionary_scaling_summary.csv"
    save_summary_results_csv(df_summary, summary_csv_path, summary_cols)
    logger.info(f"Saved summary results to {summary_csv_path}")
    
    # Generate Plots
    plot_metric_vs_dictionary_size(df_summary, "mae", "MAE (Mean Absolute Error)", "Dictionary Scaling: MAE vs Dictionary Size", "results/plots/dictionary_scaling_mae.png")
    plot_metric_vs_dictionary_size(df_summary, "rmse", "RMSE (Root Mean Square Error)", "Dictionary Scaling: RMSE vs Dictionary Size", "results/plots/dictionary_scaling_rmse.png")
    plot_metric_vs_dictionary_size(df_summary, "nmae", "Normalized MAE", "Dictionary Scaling: NMAE vs Dictionary Size", "results/plots/dictionary_scaling_nmae.png")
    plot_metric_vs_dictionary_size(df_summary, "top10_overlap", "Top-10 Overlap Ratio", "Dictionary Scaling: Top-10 Overlap vs Dictionary Size", "results/plots/dictionary_scaling_top10.png")
    plot_metric_vs_dictionary_size(df_summary, "spearman", "Spearman Rank Correlation", "Dictionary Scaling: Spearman Correlation vs Dictionary Size", "results/plots/dictionary_scaling_spearman.png")
    
    logger.info("Completed Experiment 3: Dictionary-Size Sensitivity")


if __name__ == "__main__":
    run_dictionary_scaling_experiment()
