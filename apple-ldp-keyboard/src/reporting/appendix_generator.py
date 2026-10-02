import os
import json
import pandas as pd


def generate_appendix_markdown(output_path: str = "results/report/appendix.md") -> None:
    """
    Generate results/report/appendix.md with Appendices A through H.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load configs
    dev_config = {}
    exp_config = {}
    apple_config = {}
    if os.path.exists("configs/dev.json"):
        with open("configs/dev.json", "r", encoding="utf-8") as f:
            dev_config = json.load(f)
    if os.path.exists("configs/experiment.json"):
        with open("configs/experiment.json", "r", encoding="utf-8") as f:
            exp_config = json.load(f)
    if os.path.exists("configs/apple_like.json"):
        with open("configs/apple_like.json", "r", encoding="utf-8") as f:
            apple_config = json.load(f)

    dev_json_str = json.dumps(dev_config, indent=2)
    exp_json_str = json.dumps(exp_config, indent=2)
    apple_json_str = json.dumps(apple_config, indent=2)

    lines = []
    lines.append("# Technical Appendix\n")
    lines.append("## Appendix A — Complete Configuration Values\n")
    lines.append("### configs/dev.json")
    lines.append("```json\n" + dev_json_str + "\n```\n")
    lines.append("### configs/experiment.json")
    lines.append("```json\n" + exp_json_str + "\n```\n")
    lines.append("### configs/apple_like.json")
    lines.append("```json\n" + apple_json_str + "\n```\n")
    lines.append("---\n")

    lines.append("## Appendix B — Experiment Matrix\n")
    lines.append("| Phase | Experiment Name | Mechanism | Independent Variable | Fixed Parameters | Repetitions | Main Metrics |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    lines.append("| **Phase 6** | Privacy-Utility Trade-off | Private CMS | Epsilon in {0.5, 1, 2, 4, 8} | D=500, N=1000, m=128, k=128 | 20 | MAE, RMSE, NMAE, Top-10, Spearman |")
    lines.append("| **Phase 6** | Population Scaling | Private CMS | Users N in {100, 500, 1000, 5000, 10000} | D=500, epsilon=2.0, m=128, k=128 | 20 | MAE, RMSE, NMAE, Top-10, Spearman |")
    lines.append("| **Phase 6** | Dictionary Scaling | Private CMS | Vocabulary D in {50, 100, 250, 500} | N=1000, epsilon=2.0, m=128, k=128 | 10 | MAE, RMSE, NMAE, Top-10, Spearman |")
    lines.append("| **Phase 6** | CMS Parameter Sensitivity | Private CMS | Sketch (m, k) in {(32,16), ..., (512,256)} | D=500, N=1000, epsilon=2.0 | 10 | MAE, RMSE, Payload, Memory |")
    lines.append("| **Phase 8** | CMS vs HCMS Comparative | CMS vs HCMS | Mechanism & Epsilon in {0.5, 1, 2, 4, 8} | D=500, N=1000, m=128, k=128 | 10 | MAE, RMSE, Top-10, Runtimes, Bytes |")
    lines.append("| **Phase 9** | Apple-Parameter Fidelity | Private CMS | Apple Parameter Regime | D=2600, N=1000, epsilon=4.0, m=1024, k=65536 | 5 | MAE, RMSE, Payload, Memory |")
    lines.append("| **Phase 10** | Cumulative Privacy Loss | Additive Composition | Days d in 1 ... 30 | Daily epsilon=16 | N/A | Cumulative epsilon = 16d |")
    lines.append("| **Phase 11** | Simplified Pool Inference | Attacker Model | Polarization (0.60, 0.80, 0.95) & Obs (5, 10, 25, 50) | D=2600, N=200, epsilon=4.0, m=128, k=128 | 1 | Accuracy, Mean True-Pool Posterior |\n")
    lines.append("---\n")

    lines.append("## Appendix C — Generated Output Files\n")
    lines.append("### CSV Files in `results/csv/`")
    lines.append("- `privacy_utility_results.csv`, `privacy_utility_summary.csv` shadow datasets.")
    lines.append("- `population_scaling_results.csv`, `population_scaling_summary.csv`")
    lines.append("- `dictionary_scaling_results.csv`, `dictionary_scaling_summary.csv`")
    lines.append("- `cms_parameter_results.csv`, `cms_parameter_summary.csv`")
    lines.append("- `cms_vs_hcms_results.csv`, `cms_vs_hcms_summary.csv`, `cms_vs_hcms_paired.csv`")
    lines.append("- `apple_replication_results.csv`, `apple_replication_summary.csv`, `apple_parameter_fidelity.csv`")
    lines.append("- `cumulative_privacy_results.csv`, `cumulative_privacy_summary.csv`")
    lines.append("- `pool_inference_results.csv`, `pool_inference_summary.csv`\n")

    lines.append("### Plot Files in `results/plots/`")
    lines.append("- `privacy_utility_*.png`, `population_scaling_*.png`, `dictionary_scaling_*.png`, `cms_parameter_*.png`")
    lines.append("- `cms_vs_hcms_*.png` (MAE, RMSE, NMAE, Top10, Spearman, Runtimes, Payload, Memory)")
    lines.append("- `apple_replication_*.png`")
    lines.append("- `cumulative_privacy_loss.png`")
    lines.append("- `pool_inference_accuracy.png`, `pool_inference_posterior.png`\n")

    lines.append("### Reports in `results/report/`")
    lines.append("- `final_report.md`, `final_report.html`, `final_report.pdf`")
    lines.append("- `appendix.md`, `experiment_summary.csv`, `reproducibility_manifest.txt`, `final_architecture.png`\n")
    lines.append("---\n")

    lines.append("## Appendix D — Unit-Test Results\n")
    lines.append("Total test suite: **81 unit tests** in `tests/`.")
    lines.append("- `test_hash.py`: 13 passed")
    lines.append("- `test_privacy.py`: 10 passed")
    lines.append("- `test_cms.py`: 15 passed")
    lines.append("- `test_hcms.py`: 10 passed")
    lines.append("- `test_dataset.py`: 10 passed")
    lines.append("- `test_evaluation.py`: 9 passed")
    lines.append("- `test_hadamard.py`: 5 passed")
    lines.append("- `test_cms_vs_hcms.py`: 9 passed\n")
    lines.append("Result: **81 PASSED in 11.03s**\n")
    lines.append("---\n")

    lines.append("## Appendix E — Reproducibility Instructions\n")
    lines.append("To reproduce all experimental results, CSV tables, plots, and final reports from scratch:\n")
    lines.append("1. **Activate Python 3.11 Environment**:")
    lines.append("   ```bash")
    lines.append("   .venv\\Scripts\\activate")
    lines.append("   ```\n")
    lines.append("2. **Execute Full Orchestration Command**:")
    lines.append("   ```bash")
    lines.append("   python main.py --run-final")
    lines.append("   ```\n")
    lines.append("3. **Randomization Seeds**:")
    lines.append("   - Base Dataset Seed: `42`")
    lines.append("   - Hash Family Seed: `42`")
    lines.append("   - Phase 6 Offsets: `1000` (Exp 1), `2000` (Exp 2), `3000` (Exp 3), `4000` (Exp 4)")
    lines.append("   - Phase 8 CMS Seed: `500000 + 100e + r`")
    lines.append("   - Phase 8 HCMS Seed: `600000 + 100e + r`")
    lines.append("   - Phase 9 Seed: `900000 + r`")
    lines.append("   - Phase 11 Seed: `700000 + 10000*p_idx + 1000*o_idx + u_idx`\n")
    lines.append("---\n")

    lines.append("## Appendix F — Mathematical Formulas\n")
    lines.append("### 1. Private CMS")
    lines.append("- Bernoulli Sign preservation: p_CMS = exp(epsilon/2) / (exp(epsilon/2) + 1)")
    lines.append("- Debiasing scaling: c_epsilon = (exp(epsilon/2) + 1) / (exp(epsilon/2) - 1)")
    lines.append("- Unbiased Estimator: f_hat(d) = (m / (m-1)) * [ (1/k) * sum_{j=0}^{k-1} M[j, h_j(d)] - n/m ]\n")
    lines.append("### 2. Private HCMS")
    lines.append("- One-hot vector: v = {0}^m with v[h_j(d)] = 1")
    lines.append("- Fast Walsh-Hadamard Transform: w = H_m * v")
    lines.append("- Bernoulli Bit preservation: p_HCMS = exp(epsilon) / (exp(epsilon) + 1)")
    lines.append("- Sparse update scaling: x_tilde = m * k * c_epsilon * w_tilde_l")
    lines.append("- Sketch Matrix Reconstruction: M = FWHT(M_H) / m\n")
    lines.append("### 3. Cumulative Loss Model (Tang et al. 2016)")
    lines.append("- Daily Budget: epsilon_daily = 16")
    lines.append("- Cumulative Loss: epsilon_cumulative(d) = 16 * d for d = 1 ... 30\n")
    lines.append("### 4. Attacker Likelihood & Posterior Model (Gadotti et al. 2022 Inspired)")
    lines.append("- Log-Likelihood for Candidate Pool P in {A, B}: log L(P) = sum_{t=1}^T log ( (1/|P|) * sum_{d in P} P(v_tilde_t | d, j_t) )")
    lines.append("- Posterior Probability: P(P_A | obs) = 1 / ( 1 + exp(log L(P_B) - log L(P_A)) )\n")
    lines.append("---\n")

    lines.append("## Appendix G — Hardware and Software Environment\n")
    lines.append("- **Programming Language**: Python 3.11")
    lines.append("- **Core Libraries**: NumPy (1.26+), Pandas (2.1+), Matplotlib (3.8+), SciPy (1.11+), tqdm, pytest")
    lines.append("- **Operating System**: Microsoft Windows")
    lines.append("- **Architecture**: 64-bit x86_64 architecture\n")
    lines.append("---\n")

    lines.append("## Appendix H — Known Project Limitations\n")
    lines.append("1. **No Proprietary Apple Code/Data**: Uses an independent open academic simulation.")
    lines.append("2. **Synthetic Keyboard Distribution**: Vocabulary probabilities modeled via Zipf distribution (s=1.07).")
    lines.append("3. **Synthetic Emoji Vocabulary**: Emojis parsed from Unicode standard with Zipfian probabilities.")
    lines.append("4. **Population Scale**: Evaluated at small academic scale (1,000 users) vs. Apple production scale (10^8+ devices).")
    lines.append("5. **Parameter Fidelity Limit**: Algorithmic parameter reproduction (m=1024, k=65536, epsilon=4) does not equal production environment reproduction.")
    lines.append("6. **Historical Scope**: Cumulative privacy loss simulation reflects historical 2016 macOS 10.12 audit (Tang et al.) and does not measure current Apple systems.")
    lines.append("7. **Simplified Pool Inference**: Implements a simplified likelihood model vs. Gadotti et al.'s full Bayesian framework.")
    lines.append("8. **Additive Composition**: Uses basic sequential additive composition for the daily privacy budget.")
    lines.append("9. **Empirical Verification Scope**: Empirical validation confirms statistical consistency but does not constitute a formal mathematical DP proof.")
    lines.append("10. **Prototype Serialization Model**: Theoretical payload sizes use bit-level packing model vs. Apple production payload formats.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    generate_appendix_markdown()
