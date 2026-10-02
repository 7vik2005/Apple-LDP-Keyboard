import sys
import os
import json
import argparse

from src.experiments.privacy_utility import run_privacy_utility_experiment
from src.experiments.population_scaling import run_population_scaling_experiment
from src.experiments.dictionary_scaling import run_dictionary_scaling_experiment
from src.experiments.cms_parameters import run_cms_parameters_experiment
from src.experiments.cms_vs_hcms import run_cms_vs_hcms_experiment
from src.experiments.apple_replication import run_apple_replication_experiment
from src.experiments.cumulative_privacy import run_cumulative_privacy_experiment
from src.experiments.pool_inference import run_pool_inference_experiment
from src.reporting.report_generator import generate_reports


def validate_environment():
    print("==================================================")
    print(" Apple-Inspired Local Differential Privacy Project")
    print(" Environment Validation")
    print("==================================================")

    all_passed = True

    # 1. Verify Python Version
    sys.stdout.write("Checking Python version... ")
    if sys.version_info.major == 3 and sys.version_info.minor == 11:
        print(f"PASS (Found {sys.version.split()[0]})")
    else:
        print(f"FAIL (Found {sys.version.split()[0]}, expected 3.11.x)")
        all_passed = False

    # 2. Verify Packages
    packages = ["numpy", "pandas", "matplotlib", "scipy", "tqdm", "pytest"]
    for pkg in packages:
        sys.stdout.write(f"Checking package '{pkg}'... ")
        try:
            __import__(pkg)
            print("PASS")
        except ImportError:
            print("FAIL")
            all_passed = False

    # 3. Verify Directories
    required_dirs = [
        "configs",
        "data/raw",
        "data/processed",
        "data/generated",
        "src",
        "src/data",
        "src/crypto",
        "src/cms",
        "src/hcms",
        "src/privacy",
        "src/experiments",
        "src/evaluation",
        "src/reporting",
        "tests",
        "results/csv",
        "results/plots",
        "results/logs",
        "results/report"
    ]
    for d in required_dirs:
        sys.stdout.write(f"Checking directory '{d}'... ")
        if os.path.isdir(d):
            print("PASS")
        else:
            os.makedirs(d, exist_ok=True)
            print(f"CREATED (was missing)")

    # 4. Verify Configuration Files
    configs = ["configs/dev.json", "configs/experiment.json", "configs/apple_like.json"]
    for c in configs:
        sys.stdout.write(f"Checking config '{c}'... ")
        if os.path.isfile(c):
            print("PASS")
        else:
            print("FAIL")
            all_passed = False

    print("--------------------------------------------------")
    if all_passed:
        print("Validation Result: ALL PASS")
    else:
        print("Validation Result: FAIL — please resolve failures above")
    print("==================================================")
    return all_passed


def _csv_exists(path: str) -> bool:
    """Return True if a CSV result file exists and is non-empty."""
    return os.path.isfile(path) and os.path.getsize(path) > 0


def run_final_workflow():
    print("==================================================")
    print(" EXECUTING FINAL WORKFLOW (Phases 6, 8, 9, 10, 11, 12)")
    print("==================================================")

    # Ensure all output directories exist
    for d in ["results/csv", "results/plots", "results/logs", "results/report", "data/generated"]:
        os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------
    # PHASE 6: Privacy-Utility Experiments
    # -------------------------------------------------------
    print("\n--- PHASE 6a: Privacy-Utility Experiment ---")
    if _csv_exists("results/csv/privacy_utility_summary.csv"):
        print("    [SKIP] privacy_utility_summary.csv already exists.")
    else:
        run_privacy_utility_experiment()

    print("\n--- PHASE 6b: Population Scaling Experiment ---")
    if _csv_exists("results/csv/population_scaling_summary.csv"):
        print("    [SKIP] population_scaling_summary.csv already exists.")
    else:
        run_population_scaling_experiment()

    print("\n--- PHASE 6c: Dictionary Scaling Experiment ---")
    if _csv_exists("results/csv/dictionary_scaling_summary.csv"):
        print("    [SKIP] dictionary_scaling_summary.csv already exists.")
    else:
        run_dictionary_scaling_experiment()

    print("\n--- PHASE 6d: CMS Parameter Sensitivity Experiment ---")
    if _csv_exists("results/csv/cms_parameter_summary.csv"):
        print("    [SKIP] cms_parameter_summary.csv already exists.")
    else:
        run_cms_parameters_experiment()

    # -------------------------------------------------------
    # PHASE 8: CMS vs HCMS Comparative Evaluation
    # -------------------------------------------------------
    print("\n--- PHASE 8: CMS vs HCMS Comparative Evaluation ---")
    if _csv_exists("results/csv/cms_vs_hcms_summary.csv"):
        print("    [SKIP] cms_vs_hcms_summary.csv already exists.")
    else:
        run_cms_vs_hcms_experiment()

    # -------------------------------------------------------
    # PHASE 9: Apple-Parameter Experiment
    # -------------------------------------------------------
    print("\n--- PHASE 9: Apple-Parameter Experiment ---")
    if _csv_exists("results/csv/apple_replication_summary.csv"):
        print("    [SKIP] apple_replication_summary.csv already exists.")
    else:
        run_apple_replication_experiment()

    # -------------------------------------------------------
    # PHASE 10: Cumulative Privacy Simulation
    # -------------------------------------------------------
    print("\n--- PHASE 10: Cumulative Privacy Simulation ---")
    if _csv_exists("results/csv/cumulative_privacy_summary.csv"):
        print("    [SKIP] cumulative_privacy_summary.csv already exists.")
    else:
        run_cumulative_privacy_experiment()

    # -------------------------------------------------------
    # PHASE 11: Simplified Pool Inference Experiment
    # -------------------------------------------------------
    print("\n--- PHASE 11: Simplified Pool Inference Experiment ---")
    if _csv_exists("results/csv/pool_inference_summary.csv"):
        print("    [SKIP] pool_inference_summary.csv already exists.")
    else:
        run_pool_inference_experiment()

    # -------------------------------------------------------
    # PHASE 12: Integrated Evaluation & Report Generation
    # -------------------------------------------------------
    print("\n--- PHASE 12: Integrated Evaluation & Report Generation ---")
    generate_reports()

    print("\n==================================================")
    print(" FINAL WORKFLOW COMPLETED SUCCESSFULLY")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser(
        description="Apple-Inspired Local Differential Privacy — Research Project Orchestrator"
    )
    parser.add_argument(
        "--run-final",
        action="store_true",
        help="Execute complete final workflow: Phases 6, 8, 9, 10, 11, 12 (skips already-complete phases)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate environment (Python version, packages, directories, configs)"
    )
    args = parser.parse_args()

    if args.run_final:
        validate_environment()
        run_final_workflow()
    elif args.validate:
        validate_environment()
    else:
        validate_environment()


if __name__ == "__main__":
    main()
