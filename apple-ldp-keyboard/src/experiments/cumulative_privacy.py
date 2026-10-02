import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def setup_logger():
    os.makedirs("results/logs", exist_ok=True)
    logger = logging.getLogger("phase10_cumulative_privacy")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("results/logs/phase10.log", mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def run_cumulative_privacy_experiment():
    """
    Phase 10: Historical Cumulative Privacy-Loss Simulation.

    Models the sequential additive privacy budget accumulation described in the
    historical audit of Apple's macOS 10.12 LDP implementation by Tang et al. (2017).
    Uses daily_epsilon=16 over 30 days, yielding a 30-day cumulative epsilon of 480.

    This simulation reflects the historical audit model reported by Tang et al. (2017)
    for macOS 10.12 and does NOT measure current Apple iOS/macOS privacy controls.

    Reference:
        Tang, J., Korolova, A., Bai, X., Wang, X., & Wang, X. (2017).
        Privacy loss in Apple's implementation of differential privacy on macOS 10.12.
        arXiv:1709.02753. https://arxiv.org/abs/1709.02753
    """
    logger = setup_logger()
    logger.info("Starting Phase 10: Cumulative Privacy-Loss Simulation")

    # Fixed Cumulative-Loss Parameters (Tang et al. 2017 historical audit model)
    daily_epsilon = 16
    days_count = 30

    rows = []
    for day in range(1, days_count + 1):
        cum_eps = day * daily_epsilon
        rows.append({
            "day": day,
            "daily_epsilon": daily_epsilon,
            "cumulative_epsilon": cum_eps
        })

    os.makedirs("results/csv", exist_ok=True)
    df_results = pd.DataFrame(rows)
    raw_csv_path = "results/csv/cumulative_privacy_results.csv"
    df_results.to_csv(raw_csv_path, index=False)
    logger.info(f"Saved raw cumulative privacy results: {raw_csv_path}")

    # Summary CSV
    final_cum_eps = days_count * daily_epsilon
    summary_row = {
        "days": days_count,
        "daily_epsilon": daily_epsilon,
        "final_cumulative_epsilon": final_cum_eps,
        "composition_rule": "sequential_additive_composition",
        "source_scope": "historical_Tang_et_al_2017_macOS_10_12_audit"
    }
    df_summary = pd.DataFrame([summary_row])
    summary_csv_path = "results/csv/cumulative_privacy_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved summary cumulative privacy CSV: {summary_csv_path}")

    # Plot
    os.makedirs("results/plots", exist_ok=True)
    plot_path = "results/plots/cumulative_privacy_loss.png"

    plt.figure()
    plt.plot(df_results["day"].values, df_results["cumulative_epsilon"].values, "-o")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Epsilon")
    plt.title("30-Day Cumulative Privacy Loss (Tang et al. 2017 Historical Audit Model)")
    plt.grid(True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cumulative privacy plot: {plot_path}")

    logger.info("Phase 10 cumulative privacy simulation completed successfully.")
    print("=== Phase 10 Cumulative Privacy Simulation Complete ===")
    print(f"Summary saved to: {summary_csv_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    run_cumulative_privacy_experiment()
