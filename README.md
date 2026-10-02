# Apple-Inspired Local Differential Privacy for Keyboard Data Collection

> **Implementation and Experimental Evaluation of Apple-Inspired Local Differential Privacy for Keyboard Data Collection**
> An open, reproducible, Python-based academic research project.

---

## 1. Project Overview

Modern smartphones collect keystroke and emoji usage data to improve autocorrect, emoji suggestions, and other features. This data is inherently private — it can reveal personal interests, beliefs, health conditions, and relationships. Sending raw keyboard data to a central server is unacceptable from a privacy standpoint.

**Local Differential Privacy (LDP)** solves this by adding mathematical noise to each user's data *on their own device* before it is ever transmitted. The server receives only privatized, individually uninformative records. When aggregated across thousands or millions of users, these records yield statistically accurate population-level frequency estimates — without revealing any individual's behaviour.

**Apple** publicly documented its LDP architecture for collecting emoji and word frequency data in its 2017 research paper. The core mechanisms are:
- **Private Count Mean Sketch (CMS)**: randomizes a ±1 sign vector.
- **Private Hadamard Count Mean Sketch (HCMS)**: applies a Hadamard transform before single-bit randomization, dramatically reducing the communication payload.

This project **implements both mechanisms from scratch**, evaluates them across a comprehensive parameter space, and analyses practical privacy considerations using entirely synthetic data and no proprietary Apple code or data.

### Simple Data Flow

```text
raw keyboard input (item d)
        ↓
local hash encoding  →  ±1 sign vector (CMS) or one-hot → Hadamard (HCMS)
        ↓
local randomization  →  Bernoulli noise applied per-bit (epsilon-LDP)
        ↓
privatized payload   →  transmitted to server (raw d never leaves device)
        ↓
server aggregation   →  sketch matrix M accumulated over n records
        ↓
unbiased estimation  →  f̃(d) = population-level frequency estimate
        ↓
evaluation           →  MAE, RMSE, Top-10 overlap, Spearman vs. ground truth
```

---

## 2. Problem Statement

```text
useful aggregate statistics   ←→   individual privacy
```

Aggregate keyboard frequency data is genuinely useful for:
- Training autocorrect and predictive text models
- Improving emoji suggestion relevance
- Understanding evolving language patterns

But the raw keyboard stream reveals:
- Search terms and personal messages
- Health-related queries
- Political, religious, or personal preferences

Sending raw data creates an irreconcilable privacy risk. LDP provides a formal mathematical guarantee:

> **ε-Local Differential Privacy**: A randomized mechanism M satisfies ε-LDP if for any two inputs d and d′, and any output set S:
> P[M(d) ∈ S] ≤ exp(ε) · P[M(d′) ∈ S]

This means the server cannot distinguish, with high confidence, whether a user typed item d or any other item d′.

---

## 3. Research Question

> **How does the choice of the privacy parameter (ε) in Apple's local differential privacy mechanisms affect the accuracy of keyboard word-frequency estimation, and how do repeated submissions affect the practical privacy guarantees observed in a controlled implementation?**

### Sub-questions

1. How does estimation error (MAE, RMSE, NMAE, Top-10 Overlap, Spearman) scale with ε ∈ {0.5, 1, 2, 4, 8}?
2. How does population scale (N ∈ {100, 500, 1000, 5000, 10000}) reduce randomization noise?
3. How do sketch dimensions (m, k) impact accuracy, client payload, and sketch memory?
4. What are the empirical trade-offs between Private CMS and Private HCMS under identical paired conditions?
5. How does our prototype perform under Apple's published parameter regime (ε=4, m=1024, k=65536, D=2600)?
6. What is the cumulative privacy budget trajectory under the repeated-collection model from Tang et al. (2017)?
7. How does a simplified likelihood-based adversary infer user pool preferences from repeated privatized CMS observations?

---

## 4. Research Gap

Differential privacy theory is mature and well-studied (Dwork & Roth, 2014). Apple publicly documented its LDP architecture in detail (Apple Differential Privacy Team, 2017). Independent researchers have audited Apple's historical macOS 10.12 implementation (Tang et al., 2017) and studied inference attack risks (Gadotti et al., 2022).

However, these works are scattered across theoretical papers, corporate documentation, and security audits. **This project consolidates them** into a single, reproducible, from-scratch Python implementation with controlled experimental evaluation.

> **Our contribution**: a reproducible implementation and empirical study of publicly documented Apple-inspired LDP mechanisms and their privacy–utility and practical privacy behaviour, using only synthetic data and open code.

We do **not** claim to have implemented Apple's production system, nor do we claim any new DP algorithm.

---

## 5. Objectives

1. Implement a deterministic GF(2¹²) polynomial hash family guaranteeing three-wise independence.
2. Implement Private CMS: client encoder, Bernoulli randomizer, server unbiased frequency estimator.
3. Implement Private HCMS: one-hot encoder, Fast Walsh–Hadamard Transform, single-bit randomizer, server inverse FWHT reconstruction.
4. Conduct Phase 6 experiments: privacy–utility sweep, population scaling, dictionary scaling, CMS parameter sensitivity.
5. Conduct Phase 8: fair paired CMS vs. HCMS evaluation (accuracy, runtime, payload, memory).
6. Conduct Phase 9: Apple-parameter regime evaluation (ε=4, m=1024, k=65536, D=2600).
7. Conduct Phase 10: 30-day cumulative privacy-loss simulation (historical Tang et al. model).
8. Conduct Phase 11: simplified two-pool likelihood-based inference attack.
9. Automate reproducible execution, data collection, visualization, and multi-format report generation.

---

## 6. What This Project Implements

| Component | Role |
|---|---|
| **Synthetic Word Dataset** | 500 synthetic words with Zipf(s=1.07) probabilities |
| **Synthetic Emoji Dataset** | 2,600 fully-qualified emojis from the Unicode standard |
| **Ground Truth** | Pre-privatization count and frequency baseline |
| **Three-Wise Hash Family** | GF(2¹²) quadratic polynomials; deterministic and reproducible |
| **Private CMS** | Client encoding + Bernoulli sign randomization + server aggregation + unbiased estimator |
| **Private HCMS** | One-hot → FWHT → single-bit randomization + server inverse FWHT + frequency estimator |
| **Phase 6 Experiments** | Privacy–utility, population scaling, dictionary scaling, CMS parameter sensitivity |
| **Phase 8 Experiment** | CMS vs HCMS paired comparison |
| **Phase 9 Experiment** | Apple-parameter regime fidelity |
| **Phase 10 Simulation** | Historical cumulative privacy-loss model |
| **Phase 11 Attack** | Simplified likelihood-based two-pool inference |
| **Evaluation** | MAE, RMSE, NMAE, Top-10 Overlap, Spearman, communication bytes, sketch memory, runtime |
| **Report Generation** | Automated Markdown, HTML, PDF reports + reproducibility manifest |

---

## 7. What This Project Does NOT Implement

| Out of Scope | Reason |
|---|---|
| Actual iOS keyboard | Not available; project is educational/academic |
| Proprietary Apple source code | Closed-source; not reproducible |
| Proprietary Apple dataset | Private user data; not available |
| Shuffled Federated Protocol (SFP) | Related work only; outside scope |
| Federated learning | Related work only; outside scope |
| Real user keyboard data | Not collected; synthetic data only |
| Cloud/AWS/Docker deployment | Not required for academic simulation |
| Database infrastructure | Not required |
| Machine learning models | Not required |
| Additional DP libraries | Banned; everything implemented from scratch |

---

## 8. Core Concepts

### Differential Privacy

Differential privacy (Dwork & Roth, 2014) provides a formal mathematical privacy guarantee. A mechanism M is ε-DP if changing any single individual's record changes the probability of any output by at most a factor of exp(ε).

### Local Differential Privacy

In the **local** model, each user applies the randomization mechanism on their own device. The server never sees raw data. This is a stronger privacy model than the central DP model because there is no trusted curator.

### Epsilon (ε)

The privacy parameter ε controls the privacy–utility trade-off:
- **Small ε** (e.g., 0.5) → more noise → stronger privacy → lower accuracy.
- **Large ε** (e.g., 8.0) → less noise → weaker privacy → higher accuracy.

### Privacy–Utility Trade-off

Accuracy and privacy are fundamentally opposed. As ε increases, the probability that a client reports their true encoding increases. As ε → ∞, the mechanism becomes deterministic. As ε → 0, the output is uniformly random.

### Count Mean Sketch (CMS)

CMS (Apple, 2017) is a private frequency estimation data structure. Each client:
1. Selects a random hash function index j ∈ {0, …, k-1}.
2. Computes the hash h_j(d) ∈ {0, …, m-1}.
3. Constructs a ±1 sign vector v of length m, with v[h_j(d)] = +1 and all others = -1.
4. Applies Bernoulli randomization: each element is flipped with probability 1-p where p = exp(ε/2) / (exp(ε/2) + 1).
5. Transmits (ṽ, j) — raw d never transmitted.

The server computes the unbiased frequency estimate:

```
f̃(d) = m/(m-1) · [ (1/k) · Σⱼ M[j, hⱼ(d)] - n/m ]
```

where M is the k×m sketch matrix, and n is the total record count.

### Hadamard Count Mean Sketch (HCMS)

HCMS (Apple, 2017) reduces client communication from O(m) bits to O(1) bits by:
1. Constructing a **one-hot vector** v ∈ {0,1}^m with v[h_j(d)] = 1.
2. Computing the **Fast Walsh–Hadamard Transform** w = H_m · v.
3. Sampling a random coordinate l ∈ {0, …, m-1}.
4. Applying single-bit LDP randomization: p_HCMS = exp(ε) / (exp(ε) + 1).
5. Transmitting only (bit, j, l) — 3 indices instead of m bits.

The server accumulates a sparse sketch matrix M_H, then reconstructs:

```
M = FWHT(M_H) / m
```

Frequency estimation uses the same CMS formula applied to the reconstructed M.

### Three-Wise Independent Hashing

The hash family uses quadratic polynomials over GF(2¹²):
```
h_j(x) = a_j · x² + b_j · x + c_j   (mod irreducible polynomial x¹² + x³ + 1)
```

Three-wise independence ensures that any three items hash independently, which is critical for the unbiasedness guarantees.

### Sequential Composition

Under sequential (additive) composition, if a user submits events over d days with per-day budget ε_daily, the cumulative budget is:
```
ε_cumulative = Σᵢ εᵢ = ε_daily · d
```

This is the **basic** composition theorem, which provides a worst-case upper bound.

### Pool Inference

Pool inference (Gadotti et al., 2022) asks: can an adversary determine which group of items a user prefers, from their privatized observations? Our simplified experiment tests a two-pool binary classifier using log-likelihood maximization.

---

## 9. System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    CLIENT (Local Device)                   │
│                                                           │
│  Synthetic Dataset (Word / Emoji vocabulary + Zipf probs) │
│              ↓                                            │
│  Simulated User: sample item d from Zipf distribution     │
│              ↓                                            │
│  Hash encoding: construct v (CMS: ±1 vector,              │
│                              HCMS: one-hot → FWHT)        │
│              ↓                                            │
│  Local Randomization: Bernoulli noise (ε-LDP)             │
│              ↓                                            │
│  Privatized payload: (ṽ, j) or (bit, j, l)               │
│                                                           │
│ ▓▓▓▓▓▓▓▓▓▓▓  PRIVACY BOUNDARY  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│   Raw item d never crosses this boundary                  │
└───────────────────────────────────────────────────────────┘
                          ↓ only privatized payload
┌───────────────────────────────────────────────────────────┐
│                    SERVER (Aggregator)                     │
│                                                           │
│  CMS: Dense sketch M[j, :] += k·(c_ε/2·ṽ + ½·1)         │
│  HCMS: Sparse M_H[j, l] += m·k·c_ε·bit                  │
│              ↓  (after all n records)                     │
│  HCMS only: M = FWHT(M_H) / m  (inverse reconstruction)  │
│              ↓                                            │
│  Frequency Estimator:                                     │
│    f̃(d) = m/(m-1) · [(1/k)·Σⱼ M[j,hⱼ(d)] - n/m]       │
│              ↓                                            │
│  Evaluation: compare f̃(d) vs. ground truth f(d)         │
└───────────────────────────────────────────────────────────┘
```

---

## 10. Detailed CMS Workflow

| Step | What Happens | Formula |
|---|---|---|
| 1. Input | User has item d (encoded as integer item_id) | — |
| 2. Hash selection | Sample j ~ Uniform{0, …, k-1} | — |
| 3. Hash evaluation | Compute h_j(d) ∈ {0, …, m-1} | h_j(d) = (a_j·d² + b_j·d + c_j) mod GF(2¹²) masked to m |
| 4. Vector construction | v ∈ {-1,+1}^m; v[h_j(d)] = +1, others = -1 | — |
| 5. Randomization | Each v_l independently: keep with prob p, flip with prob 1-p | p_CMS = exp(ε/2) / (exp(ε/2) + 1) |
| 6. Transmit | Send (ṽ, j) — raw d never transmitted | — |
| 7. Server update | Row j of M: M[j, :] += k·(c_ε/2·ṽ + ½·**1**) | c_ε = (exp(ε/2)+1) / (exp(ε/2)-1) |
| 8. Estimation | Apply unbiased estimator to M | f̃(d) = m/(m-1)·[(1/k)·Σⱼ M[j,hⱼ(d)] - n/m] |

---

## 11. Detailed HCMS Workflow

| Step | What Happens |
|---|---|
| 1. Input | User has item d |
| 2. Hash selection | Sample j ~ Uniform{0, …, k-1} |
| 3. One-hot vector | v ∈ {0,1}^m; v[h_j(d)] = 1, others = 0 |
| 4. Hadamard transform | w = H_m · v (computed via FWHT in O(m log m)) |
| 5. Coordinate sampling | Sample l ~ Uniform{0, …, m-1} |
| 6. Single-bit LDP | bit = randomize(w_l, ε); p_HCMS = exp(ε)/(exp(ε)+1) |
| 7. Transmit | Send only (bit, j, l) — **3 integers instead of m bits** |
| 8. Server update | M_H[j, l] += m·k·c_ε·bit; c_ε_HCMS = (exp(ε)+1)/(exp(ε)-1) |
| 9. Reconstruction | After n records: M = FWHT(M_H) / m  (row-wise) |
| 10. Estimation | Same formula as CMS applied to reconstructed M |

**Why HCMS uses less communication**: CMS must transmit the entire m-dimensional randomized vector (e.g., 128 bits = 16 bytes) plus hash index. HCMS transmits only one bit plus two index integers. For m=128, k=128: CMS = 17 bytes per record; HCMS = 2 bytes per record — an 88% reduction.

---

## 12. Dataset Design

| Property | Value |
|---|---|
| Word vocabulary size | D = 500 |
| Emoji vocabulary size | D = 2,600 |
| Zipf exponent | s = 1.07 |
| Events per user | 20 |
| Primary experiment population | N = 1,000 |
| Population scaling range | N ∈ {100, 500, 1000, 5000, 10000} |

- **Word vocabulary**: synthetic tokens `word_0`, `word_1`, …, `word_499` with Zipfian probabilities.
- **Emoji vocabulary**: 2,600 fully-qualified emojis parsed from the official [Unicode emoji-test.txt](https://www.unicode.org/Public/emoji/latest/emoji-test.txt) dataset, assigned Zipfian probabilities.
- **Events**: each simulated user independently samples 20 items from the vocabulary according to its probability distribution.
- **Ground truth**: exact per-item event counts computed from the sampled events, independently of the LDP pipeline.

**Why synthetic data?** Real keyboard data is private. Generating synthetic data from a Zipfian model produces a realistic skewed distribution while maintaining full reproducibility and ensuring no personal data is handled.

> **Important**: The emoji probabilities used here are synthetically generated from a Zipf model. They do not represent Apple's actual emoji usage statistics.

---

## 13. Experimental Design

| Phase | Experiment | Mechanism | Independent Variable | Fixed Parameters | Repetitions |
|---|---|---|---|---|---|
| 6 | Privacy–Utility | Private CMS | ε ∈ {0.5, 1, 2, 4, 8} | D=500, N=1000, m=128, k=128 | 20 |
| 6 | Population Scaling | Private CMS | N ∈ {100, 500, 1000, 5000, 10000} | D=500, ε=2, m=128, k=128 | 20 |
| 6 | Dictionary Scaling | Private CMS | D ∈ {50, 100, 250, 500} | N=1000, ε=2, m=128, k=128 | 10 |
| 6 | CMS Parameters | Private CMS | (m,k) ∈ {(32,16),(64,32),(128,64),(256,128),(512,256)} | D=500, N=1000, ε=2 | 10 |
| 8 | CMS vs HCMS | CMS & HCMS | Mechanism × ε ∈ {0.5, 1, 2, 4, 8} | D=500, N=1000, m=128, k=128 | 10 |
| 9 | Apple-Parameter | Private CMS | Fixed Apple regime | D=2600, N=1000, ε=4, m=1024, k=65536 | 5 |
| 10 | Cumulative Privacy | Additive Composition | Days d ∈ 1 … 30 | ε_daily=16 | N/A |
| 11 | Pool Inference | Likelihood Attack | Polarization × Observations | D=2600, ε=4, m=128, k=128, N=200 | 1 |

---

## 14. Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | (1/D) Σ |f̃(d) - f(d)| | Mean per-item absolute estimation error |
| RMSE | √[(1/D) Σ (f̃(d) - f(d))²] | Root mean squared error; penalizes large deviations |
| NMAE | MAE / n | Normalized MAE (scale-independent) |
| Top-10 Overlap | |True Top-10 ∩ Est. Top-10| / 10 | Fraction of true top-10 items correctly identified |
| Spearman ρ | rank correlation | Rank-order agreement between true and estimated frequencies |
| Communication bytes | ⌈(m + ⌈log₂k⌉) / 8⌉ (CMS), ⌈(1 + ⌈log₂k⌉ + ⌈log₂m⌉) / 8⌉ (HCMS) | Theoretical payload size per client record |
| Sketch memory | k × m × 8 bytes | Server sketch matrix allocation (float64) |
| Runtime | Wall-clock seconds | Client and server processing time |

---

## 15. Phase-by-Phase Implementation

| Phase | Purpose | Key Implementation | Output |
|---|---|---|---|
| 1 | Project setup | Directory structure, configs, requirements | Repository skeleton |
| 2 | Synthetic dataset | Zipfian word/emoji vocabulary, event generation, ground truth | `data/generated/*.csv` |
| 3 | Hash family | GF(2¹²) arithmetic, three-wise polynomial hash family | `src/crypto/hash_family.py` |
| 4 | Private CMS | Client encoder, Bernoulli randomizer, server aggregator, estimator | `src/cms/` |
| 5 | CMS validation | Unit tests, Monte Carlo unbiasedness, reproducibility | `tests/test_cms.py` |
| 6 | Experiments 1–4 | Privacy–utility, population, dictionary, CMS parameters | `results/csv/`, `results/plots/` |
| 7 | Private HCMS | FWHT, one-hot client, sparse server, inverse FWHT | `src/hcms/` |
| 8 | CMS vs HCMS | Paired comparative evaluation | `results/csv/cms_vs_hcms_*.csv` |
| 9 | Apple parameters | Apple-regime parameter fidelity evaluation | `results/csv/apple_replication_*.csv` |
| 10 | Cumulative privacy | Historical budget accumulation simulation | `results/csv/cumulative_privacy_*.csv` |
| 11 | Pool inference | Simplified likelihood-based two-pool attack | `results/csv/pool_inference_*.csv` |
| 12 | Report generation | Automated Markdown, HTML, PDF reports + appendix + manifest | `results/report/` |

All phases: **COMPLETED**.

---

## 16. Configuration Reference

### dev.json — Development / Quick Testing

```json
{
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
```

| Parameter | Meaning |
|---|---|
| `seed` | Master random seed for reproducibility |
| `dictionary_size` | Number of vocabulary items |
| `num_users` | Simulated user population |
| `epsilon` | Privacy parameter ε |
| `cms.m` | Sketch width (number of buckets) |
| `cms.k` | Number of hash functions |
| `repetitions` | Number of experimental repetitions |

### experiment.json — Standard Experiments

```json
{
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
```

### apple_like.json — Apple-Parameter Regime

```json
{
  "seed": 42,
  "dictionary_size": 2600,
  "epsilon": 4,
  "cms": {
    "m": 1024,
    "k": 65536
  },
  "repetitions": 5
}
```

These parameters match Apple's publicly documented Popular Emojis configuration. Our population (N=1000) and synthetic data differ from Apple's production scale.

### Fixed Experimental Constants

| Constant | Value | Used In |
|---|---|---|
| Events per user | 20 | All experiments |
| Zipf exponent | 1.07 | All datasets |
| Phase 8 repetitions | 10 | CMS vs HCMS |
| Phase 10 daily epsilon | 16 | Cumulative privacy |
| Phase 10 days | 30 | Cumulative privacy |
| Phase 11 epsilon | 4.0 | Pool inference |
| Phase 11 m | 128 | Pool inference |
| Phase 11 k | 128 | Pool inference |
| Phase 11 Pool A | items 0–49 (first 50 emojis) | Pool inference |
| Phase 11 Pool B | items 50–99 (next 50 emojis) | Pool inference |
| Phase 11 users per condition | 200 | Pool inference |
| Phase 11 polarization values | 0.60, 0.80, 0.95 | Pool inference |
| Phase 11 observation counts | 5, 10, 25, 50 | Pool inference |

---

## 17. Project Directory Structure

```
apple-ldp-keyboard/
│
├── README.md                        ← This file
├── requirements.txt                 ← Fixed dependency versions
├── .gitignore                       ← Excludes caches, results, generated data
├── main.py                          ← Primary entry point
│
├── configs/                         ← Experiment configurations
│   ├── dev.json                     ← Quick development config (D=50)
│   ├── experiment.json              ← Standard experiment config (D=500)
│   └── apple_like.json              ← Apple-parameter regime (D=2600)
│
├── data/                            ← Data directories (generated at runtime)
│   ├── raw/                         ← Reserved for raw inputs (not used)
│   ├── processed/                   ← Reserved for processed data
│   └── generated/                   ← Generated vocabularies and events
│
├── src/                             ← All source code
│   ├── data/                        ← Dataset generation
│   │   ├── dataset_generator.py     ← Event generation, ground truth
│   │   ├── word_loader.py           ← Synthetic word vocabulary
│   │   └── emoji_loader.py          ← Unicode emoji vocabulary
│   ├── crypto/                      ← Hash family
│   │   └── hash_family.py           ← GF(2¹²) three-wise independent hashes
│   ├── cms/                         ← Private CMS implementation
│   │   ├── client.py                ← Client encoder + Bernoulli randomizer
│   │   ├── server.py                ← Server aggregator
│   │   └── estimator.py             ← Unbiased frequency estimator
│   ├── hcms/                        ← Private HCMS implementation
│   │   ├── hadamard.py              ← FWHT / inverse FWHT
│   │   ├── client.py                ← One-hot + FWHT + single-bit randomizer
│   │   └── server.py                ← Sparse accumulator + FWHT reconstruction
│   ├── privacy/                     ← Privacy utilities
│   │   ├── randomizer.py            ← CMS and HCMS Bernoulli randomization
│   │   └── accountant.py            ← Module docstring (accounting is in experiments)
│   ├── experiments/                 ← All experiment scripts
│   │   ├── privacy_utility.py       ← Phase 6a: epsilon sweep
│   │   ├── population_scaling.py    ← Phase 6b: population sweep
│   │   ├── dictionary_scaling.py    ← Phase 6c: dictionary size sweep
│   │   ├── cms_parameters.py        ← Phase 6d: (m, k) parameter sweep
│   │   ├── cms_vs_hcms.py           ← Phase 8: paired CMS vs. HCMS
│   │   ├── apple_replication.py     ← Phase 9: Apple-parameter regime
│   │   └── cumulative_privacy.py    ← Phase 10: historical privacy-loss model
│   │   └── pool_inference.py        ← Phase 11: simplified pool inference attack
│   ├── evaluation/                  ← Metrics, plots, tables
│   │   ├── metrics.py               ← MAE, RMSE, NMAE, Top-10, Spearman, bytes
│   │   ├── plots.py                 ← All plot functions
│   │   └── tables.py                ← CSV saving utilities
│   └── reporting/                   ← Report generation
│       ├── report_generator.py      ← Phase 12: Markdown, HTML, PDF reports
│       └── appendix_generator.py    ← Appendix generation
│
├── tests/                           ← Complete test suite (81 tests)
│   ├── test_hash.py                 ← GF(2¹²) arithmetic and hash family
│   ├── test_cms.py                  ← CMS client, server, estimator
│   ├── test_hcms.py                 ← HCMS client, server, FWHT
│   ├── test_hadamard.py             ← FWHT properties
│   ├── test_privacy.py              ← Randomization probabilities
│   ├── test_cms_vs_hcms.py          ← Comparison experiment tests
│   ├── test_dataset.py              ← Dataset generation tests
│   └── test_evaluation.py           ← Metrics and plotting tests
│
└── results/                         ← All generated outputs (gitignored)
    ├── csv/                         ← All experiment data CSV files
    ├── plots/                       ← All generated plots (PNG)
    ├── logs/                        ← Experiment execution logs
    └── report/                      ← Final report files
```

---

## 18. Installation

### Prerequisites

- **Python 3.11** exactly (enforced at runtime)
- **pip** package manager

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Verify Installation

```bash
python main.py
```

This runs environment validation and prints PASS/FAIL for each component.

---

## 19. Running the Project

### Validate Environment Only

```bash
python main.py --validate
```

### Generate Dataset Manually (optional)

```bash
python -m src.data.dataset_generator --config configs/experiment.json
```

### Run the Full Test Suite

```bash
pytest
```

All 81 tests should pass. Tests validate:
- GF(2¹²) field arithmetic
- Three-wise hash family properties
- CMS client, server, and estimator correctness
- HCMS one-hot vector, FWHT, and reconstruction
- Randomization probability formulas
- Experiment schema correctness
- Reproducibility under fixed seeds

### Run the Complete Final Project

```bash
python main.py --run-final
```

This is the **primary command** for reproducing all results. It:
1. Validates the environment
2. Runs Phase 6 experiments (if not already complete)
3. Runs Phase 8 CMS vs HCMS comparison (if not already complete)
4. Runs Phase 9 Apple-parameter experiment (if not already complete)
5. Runs Phase 10 cumulative privacy simulation (if not already complete)
6. Runs Phase 11 pool inference experiment (if not already complete)
7. Generates Phase 12 reports (Markdown, HTML, PDF, appendix, manifest)

Already-completed phases are detected by the presence of their summary CSV and skipped automatically.

---

## 20. Output Files

### CSV Results (`results/csv/`)

| File | Contents |
|---|---|
| `privacy_utility_results.csv` | Per-repetition Phase 6 privacy–utility results |
| `privacy_utility_summary.csv` | Mean ± std summary by epsilon |
| `population_scaling_results.csv` | Per-repetition population scaling results |
| `population_scaling_summary.csv` | Mean ± std summary by user count |
| `dictionary_scaling_results.csv` | Per-repetition dictionary scaling results |
| `dictionary_scaling_summary.csv` | Mean ± std summary by dictionary size |
| `cms_parameter_results.csv` | Per-repetition CMS parameter sweep results |
| `cms_parameter_summary.csv` | Mean ± std summary by (m, k) configuration |
| `cms_vs_hcms_results.csv` | Per-repetition CMS and HCMS results |
| `cms_vs_hcms_summary.csv` | Mean ± std summary by mechanism and epsilon |
| `cms_vs_hcms_paired.csv` | Paired differences between CMS and HCMS |
| `apple_replication_results.csv` | Per-repetition Apple-parameter results |
| `apple_replication_summary.csv` | Summary statistics for Apple-parameter regime |
| `apple_parameter_fidelity.csv` | Parameter-by-parameter fidelity check |
| `cumulative_privacy_results.csv` | Daily cumulative epsilon values |
| `cumulative_privacy_summary.csv` | Summary with composition rule and source scope |
| `pool_inference_results.csv` | Per-user pool inference predictions |
| `pool_inference_summary.csv` | Accuracy and posterior by polarization and observations |

### Plots (`results/plots/`)

Privacy–utility, population scaling, dictionary scaling, CMS parameter, CMS vs HCMS (accuracy, runtime, communication, memory), Apple-parameter, cumulative privacy loss, pool inference accuracy, pool inference posterior.

### Logs (`results/logs/`)

Per-phase execution logs with timestamps: `phase6.log`, `phase8.log`, `phase9.log`, `phase10.log`, `phase11.log`.

### Reports (`results/report/`)

| File | Format | Description |
|---|---|---|
| `final_report.md` | Markdown | Full structured report |
| `final_report.html` | HTML | Self-contained styled report |
| `final_report.pdf` | PDF | Print-ready report (via Matplotlib PdfPages) |
| `appendix.md` | Markdown | Extended configuration and methodology appendix |
| `experiment_summary.csv` | CSV | All experiments with status and output files |
| `reproducibility_manifest.txt` | Text | Seeds, platform, command, directory structure |
| `final_architecture.png` | PNG | System architecture diagram |

---

## 21. Reproducibility

This project is designed for **complete reproducibility**:

| Reproducibility Mechanism | Details |
|---|---|
| Master seed | `seed = 42` in all config files |
| Per-experiment seed streams | Deterministic offset formula: `100000 + offset + repetition` |
| Hash family seeds | Always `seed = 42` unless explicitly varied |
| Dataset generation | Pure function of config + seed; no filesystem caching required |
| Experiment structure | Configuration-driven; no hidden notebook state |
| Floating-point determinism | NumPy default_rng (PCG64) with explicit seeds |
| Result provenance | Each CSV tagged with experiment name, parameters, repetition |
| Single command | `python main.py --run-final` reproduces everything |

---

## 22. Results Interpretation Guide

| Metric | Higher is… | Lower is… | Notes |
|---|---|---|---|
| MAE | Worse (more error) | Better (less error) | Main accuracy metric |
| RMSE | Worse | Better | Emphasizes large errors more than MAE |
| NMAE | Worse | Better | Scale-normalised; comparable across populations |
| Top-10 Overlap | Better (more recovered) | Worse | Ranges [0, 1]; 1.0 = perfect top-10 recovery |
| Spearman ρ | Better (closer to 1) | Worse (closer to -1) | Rank-order agreement; 1.0 = perfect |
| Communication bytes | More bandwidth | Less bandwidth | HCMS is far smaller than CMS |
| Sketch memory | More server RAM | Less server RAM | Same for CMS and HCMS given same (m, k) |
| Client runtime | Slower | Faster | HCMS includes FWHT computation |
| Server runtime | Slower | Faster | HCMS includes finalization step |
| Cumulative ε | More privacy loss | Less privacy loss | Larger ε = weaker privacy |
| Pool inference accuracy | Attacker succeeds more | Attacker fails more | Baseline = 0.50 (random guessing) |

> **Important**: Do not interpret any single configuration as "best." The appropriate configuration depends on the deployment context, population size, dictionary size, and acceptable privacy–utility trade-off.

---

## 23. Limitations

1. **Synthetic data**: Vocabulary probabilities are Zipfian (s=1.07); not real user data.
2. **Small population**: Maximum N=10,000; Apple operates at 10⁸+ device scale.
3. **No proprietary Apple implementation**: Algorithm parameters reproduced; implementation differs.
4. **No production environment reproduction**: Our Phase 9 experiment uses academic-scale simulation only.
5. **Historical Tang scope**: Phase 10 models the historical macOS 10.12 audit (Tang et al., 2017) and does not measure current Apple systems.
6. **Simplified pool inference**: Phase 11 implements a simplified likelihood model, not the full Bayesian framework from Gadotti et al. (2022).
7. **Basic additive composition**: Phase 10 uses sequential additive composition, not advanced composition or Rényi DP.
8. **Empirical verification**: Statistical tests confirm consistency but do not constitute formal mathematical DP proofs.
9. **Theoretical payload model**: Bit-level packing; actual production payload formats differ.
10. **Emoji probabilities**: Synthetically generated; not Apple's actual emoji usage statistics.

---

## 24. Security and Privacy Considerations

- **No real personal data** was collected, processed, or stored at any point.
- **Raw item IDs** never cross the simulated privacy boundary; only privatized vectors and single bits are transmitted.
- **Attack experiments** (Phase 11) use entirely synthetic data for academic evaluation of LDP mechanisms.
- This implementation is **educational and research-oriented** and is not suitable for production deployment.

---

## 25. Academic Positioning

> This project's contribution is: **a reproducible implementation and empirical study of publicly documented Apple-inspired LDP mechanisms (CMS and HCMS) and their privacy–utility and practical privacy behaviour under controlled conditions.**

We do **not** claim:
- A new differential privacy algorithm
- Reproduction of Apple's proprietary production system
- Access to or use of real Apple user data
- A formal security proof for any of the mechanisms

---

## 26. References

- **Apple Differential Privacy Team** (2017). *Learning with privacy at scale*. Apple Machine Learning Research. https://machinelearning.apple.com/research/learning-with-privacy-at-scale

- **Apple** (2017). *Differential Privacy Overview*. https://www.apple.com/privacy/docs/Differential_Privacy_Overview.pdf

- **Dwork, C., & Roth, A.** (2014). *The algorithmic foundations of differential privacy*. Foundations and Trends in Theoretical Computer Science, 9(3–4), 211–407. https://doi.org/10.1561/0400000042

- **Gadotti, A., Houssiau, F., Annamalai, M. S. M. S., & de Montjoye, Y.-A.** (2022). Pool inference attacks on local differential privacy: Quantifying the privacy guarantees of Apple's Count Mean Sketch in practice. *Proceedings of the 31st USENIX Security Symposium*, 501–518. https://www.usenix.org/conference/usenixsecurity22/presentation/gadotti

- **Hard, A., Rao, K., Mathews, R., Ramaswamy, S., Beaufays, F., Augenstein, S., Eichner, H., Kiddon, C., & Ramage, D.** (2018). Federated learning for mobile keyboard prediction. *arXiv preprint arXiv:1811.03604*. https://arxiv.org/abs/1811.03604

- **Tang, J., Korolova, A., Bai, X., Wang, X., & Wang, X.** (2017). Privacy loss in Apple's implementation of differential privacy on macOS 10.12. *arXiv preprint arXiv:1709.02753*. https://arxiv.org/abs/1709.02753

- **Unicode Consortium**. Emoji data files. https://www.unicode.org/Public/emoji/latest/emoji-test.txt

---

## 27. Quick Start

Clone or extract the project, install dependencies, and run:

```bash
python main.py --run-final
```

This single command reproduces every experiment, generates all plots, and produces the complete final report in `results/report/`.

All 81 unit tests can be validated separately:

```bash
pytest
```

Expected output:

```
81 passed in ~9s
```
