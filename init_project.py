import os
import json

base_dir = r"c:\Users\Satvik\OneDrive\Desktop\Data Privacy Case Study\apple-ldp-keyboard"

files_to_create = [
    ".gitignore",
    "main.py",
    "configs/dev.json",
    "configs/experiment.json",
    "configs/apple_like.json",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/dataset_generator.py",
    "src/data/word_loader.py",
    "src/data/emoji_loader.py",
    "src/crypto/__init__.py",
    "src/crypto/hash_family.py",
    "src/cms/__init__.py",
    "src/cms/client.py",
    "src/cms/server.py",
    "src/cms/estimator.py",
    "src/hcms/__init__.py",
    "src/hcms/hadamard.py",
    "src/hcms/client.py",
    "src/hcms/server.py",
    "src/hcms/estimator.py",
    "src/privacy/__init__.py",
    "src/privacy/randomizer.py",
    "src/privacy/accountant.py",
    "src/experiments/__init__.py",
    "src/experiments/privacy_utility.py",
    "src/experiments/population_scaling.py",
    "src/experiments/dictionary_scaling.py",
    "src/experiments/cms_parameters.py",
    "src/experiments/apple_replication.py",
    "src/experiments/cumulative_privacy.py",
    "src/evaluation/__init__.py",
    "src/evaluation/metrics.py",
    "src/evaluation/plots.py",
    "src/evaluation/tables.py",
    "src/reporting/__init__.py",
    "src/reporting/report_generator.py",
    "src/reporting/appendix_generator.py",
    "tests/test_hash.py",
    "tests/test_cms.py",
    "tests/test_hcms.py",
    "tests/test_privacy.py",
    "tests/test_hadamard.py",
    "results/csv/.gitkeep",
    "results/plots/.gitkeep",
    "results/logs/.gitkeep",
    "results/report/.gitkeep",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/generated/.gitkeep"
]

for file_path in files_to_create:
    full_path = os.path.join(base_dir, file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if not os.path.exists(full_path):
        with open(full_path, "w", encoding="utf-8") as f:
            if full_path.endswith(".py"):
                f.write('"""\nModule docstring placeholder.\n"""\n')
            else:
                f.write("")

# Write configs
dev_config = {
  "seed": 42,
  "dictionary_size": 50,
  "num_users": 1000,
  "epsilon": 2.0,
  "cms": {
    "m": 32,
    "k": 16
  },
  "repetitions": 3
}

exp_config = {
  "seed": 42,
  "dictionary_size": 500,
  "epsilon_values": [0.5, 1, 2, 4, 8],
  "user_counts": [100, 500, 1000, 5000, 10000],
  "cms": {
    "m": 128,
    "k": 128
  },
  "repetitions": 20
}

apple_config = {
  "seed": 42,
  "dictionary_size": 2600,
  "epsilon": 4,
  "cms": {
    "m": 1024,
    "k": 65536
  },
  "repetitions": 5
}

with open(os.path.join(base_dir, "configs", "dev.json"), "w", encoding="utf-8") as f:
    json.dump(dev_config, f, indent=2)
with open(os.path.join(base_dir, "configs", "experiment.json"), "w", encoding="utf-8") as f:
    json.dump(exp_config, f, indent=2)
with open(os.path.join(base_dir, "configs", "apple_like.json"), "w", encoding="utf-8") as f:
    json.dump(apple_config, f, indent=2)

print("Files created.")
