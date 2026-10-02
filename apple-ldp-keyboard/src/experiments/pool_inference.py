import os
import math
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.emoji_loader import load_emoji_vocabulary
from src.crypto.hash_family import ThreeWiseHashFamily
from src.cms.client import CMSClient
from scipy.special import logsumexp


def setup_logger():
    os.makedirs("results/logs", exist_ok=True)
    logger = logging.getLogger("phase11_pool_inference")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("results/logs/phase11.log", mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def run_pool_inference_experiment():
    logger = setup_logger()
    logger.info("Starting Phase 11: Simplified Pool-Inference Experiment")

    # Fixed Parameters
    epsilon = 4.0
    m = 128
    k = 128
    hash_seed = 42
    total_users_per_cond = 200  # 100 A-preferring, 100 B-preferring
    polarization_values = [0.60, 0.80, 0.95]
    observation_counts = [5, 10, 25, 50]

    # Load 2600 emoji vocabulary
    emoji_df = load_emoji_vocabulary(2600, s=1.07)
    item_ids = emoji_df['item_id'].values

    # Pool definitions per Section 11.2
    pool_A_items = item_ids[0:50]      # first 50
    pool_B_items = item_ids[50:100]    # next 50
    neutral_items = item_ids[100:2600] # remaining 2500

    assert len(pool_A_items) == 50
    assert len(pool_B_items) == 50
    assert len(neutral_items) == 2500

    # Neutral Zipf distribution normalized over neutral items
    neutral_probs = emoji_df['probability'].values[100:2600]
    neutral_probs = neutral_probs / np.sum(neutral_probs)

    # Hash family shared across experiment
    hash_family = ThreeWiseHashFamily(k=k, m=m, seed=hash_seed)

    # Precompute log-likelihood constants for single sign observation
    p_sign = math.exp(epsilon / 2.0) / (math.exp(epsilon / 2.0) + 1.0)
    log_p_match = math.log(p_sign)
    log_p_mismatch = math.log(1.0 - p_sign)

    raw_results = []
    summary_rows = []

    for p_idx, pol in enumerate(polarization_values):
        for o_idx, T in enumerate(observation_counts):
            logger.info(f"Running pool inference for polarization={pol}, observations={T}")

            correct_count = 0
            true_pool_posteriors = []

            for u_idx in range(total_users_per_cond):
                # 100 A-preferring (0..99), 100 B-preferring (100..199)
                true_pool = "A" if u_idx < 100 else "B"

                # Deterministic seed formula per Section 11.8
                user_seed = 700000 + p_idx * 10000 + o_idx * 1000 + u_idx
                user_rng = np.random.default_rng(user_seed)

                # Construct item sampling distribution for user
                p_pref = pol
                p_alt = (1.0 - pol) / 2.0
                p_neu = (1.0 - pol) / 2.0

                if true_pool == "A":
                    p_A_tot, p_B_tot = p_pref, p_alt
                else:
                    p_A_tot, p_B_tot = p_alt, p_pref

                # Probability vector over all 2600 items
                item_sample_probs = np.zeros(2600, dtype=np.float64)
                item_sample_probs[0:50] = p_A_tot / 50.0
                item_sample_probs[50:100] = p_B_tot / 50.0
                item_sample_probs[100:2600] = p_neu * neutral_probs

                assert np.isclose(np.sum(item_sample_probs), 1.0)

                # Client privatizer stream
                client = CMSClient(epsilon=epsilon, hash_family=hash_family, rng=user_rng)

                # Sample T user items and generate privatized observations
                user_items = user_rng.choice(item_ids, size=T, p=item_sample_probs)
                records = [client.privatize(int(item)) for item in user_items]

                # Attacker likelihood computation
                log_L_A = 0.0
                log_L_B = 0.0

                for rec in records:
                    v_tilde = np.array(rec["vector"], dtype=np.int32)
                    j = rec["hash_index"]

                    # Evaluate log-likelihood for each candidate item in Pool A
                    log_p_obs_given_d_A = []
                    for d in pool_A_items:
                        h = hash_family.hash(d, j)
                        # Position h has target sign +1, all others -1
                        matches = np.count_nonzero(v_tilde[h] == 1) + np.count_nonzero(v_tilde[np.arange(m) != h] == -1)
                        mismatches = m - matches
                        log_prob = matches * log_p_match + mismatches * log_p_mismatch
                        log_p_obs_given_d_A.append(log_prob)

                    # Marginalize over Pool A (equal prior 1/50 per item)
                    log_p_obs_given_A = logsumexp(log_p_obs_given_d_A) - math.log(50.0)
                    log_L_A += log_p_obs_given_A

                    # Evaluate log-likelihood for each candidate item in Pool B
                    log_p_obs_given_d_B = []
                    for d in pool_B_items:
                        h = hash_family.hash(d, j)
                        matches = np.count_nonzero(v_tilde[h] == 1) + np.count_nonzero(v_tilde[np.arange(m) != h] == -1)
                        mismatches = m - matches
                        log_prob = matches * log_p_match + mismatches * log_p_mismatch
                        log_p_obs_given_d_B.append(log_prob)

                    log_p_obs_given_B = logsumexp(log_p_obs_given_d_B) - math.log(50.0)
                    log_L_B += log_p_obs_given_B

                # Predict pool via maximum likelihood
                predicted_pool = "A" if log_L_A > log_L_B else "B"
                is_correct = (predicted_pool == true_pool)

                # Posterior probability calculation using log-sum-exp
                log_post_A = log_L_A - logsumexp([log_L_A, log_L_B])
                log_post_B = log_L_B - logsumexp([log_L_A, log_L_B])

                post_prob_true = math.exp(log_post_A) if true_pool == "A" else math.exp(log_post_B)

                if is_correct:
                    correct_count += 1
                true_pool_posteriors.append(post_prob_true)

                raw_results.append({
                    "polarization": pol,
                    "observations": T,
                    "user_id": u_idx,
                    "true_pool": true_pool,
                    "predicted_pool": predicted_pool,
                    "correct": int(is_correct),
                    "posterior_probability_true_pool": post_prob_true
                })

            acc = float(correct_count / total_users_per_cond)
            mean_post = float(np.mean(true_pool_posteriors))

            summary_rows.append({
                "polarization": pol,
                "observations": T,
                "users": total_users_per_cond,
                "epsilon": float(epsilon),
                "m": m,
                "k": k,
                "accuracy": acc,
                "mean_posterior_true_pool": mean_post,
                "random_guess_baseline": 0.5
            })

    # Save Output CSVs per Section 11.14 & 11.15
    os.makedirs("results/csv", exist_ok=True)
    df_raw = pd.DataFrame(raw_results)
    raw_csv_path = "results/csv/pool_inference_results.csv"
    df_raw.to_csv(raw_csv_path, index=False)
    logger.info(f"Saved raw pool inference results: {raw_csv_path}")

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = "results/csv/pool_inference_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved summary pool inference CSV: {summary_csv_path}")

    # Generate Plots per Section 11.16
    os.makedirs("results/plots", exist_ok=True)

    # Plot 1 — Pool Inference Accuracy
    plt.figure()
    for pol in polarization_values:
        sub = df_summary[df_summary["polarization"] == pol].sort_values("observations")
        plt.plot(sub["observations"].values, sub["accuracy"].values, "-o", label=f"Polarization = {pol:.2f}")
    plt.axhline(y=0.5, color="red", linestyle="--", label="Random Baseline (0.50)")
    plt.xlabel("Observations (T)")
    plt.ylabel("Inference Accuracy")
    plt.title("Simplified Pool Inference Accuracy vs Observations")
    plt.ylim((0.0, 1.05))
    plt.legend()
    plt.grid(True)
    acc_plot_path = "results/plots/pool_inference_accuracy.png"
    plt.savefig(acc_plot_path, bbox_inches="tight")
    plt.close()

    # Plot 2 — Mean Posterior Probability of True Pool
    plt.figure()
    for pol in polarization_values:
        sub = df_summary[df_summary["polarization"] == pol].sort_values("observations")
        plt.plot(sub["observations"].values, sub["mean_posterior_true_pool"].values, "-s", label=f"Polarization = {pol:.2f}")
    plt.axhline(y=0.5, color="red", linestyle="--", label="Random Baseline (0.50)")
    plt.xlabel("Observations (T)")
    plt.ylabel("Mean Posterior Probability of True Pool")
    plt.title("True Pool Mean Posterior Probability vs Observations")
    plt.ylim((0.0, 1.05))
    plt.legend()
    plt.grid(True)
    post_plot_path = "results/plots/pool_inference_posterior.png"
    plt.savefig(post_plot_path, bbox_inches="tight")
    plt.close()

    logger.info("Phase 11 pool inference experiment completed successfully.")
    print("=== Phase 11 Simplified Pool Inference Complete ===")
    print(f"Summary saved to: {summary_csv_path}")
    print(f"Plots saved to: {acc_plot_path} and {post_plot_path}")


if __name__ == "__main__":
    run_pool_inference_experiment()
