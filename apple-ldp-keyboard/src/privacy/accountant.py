"""
Privacy Accountant — Sequential Additive Composition.

This module is reserved for privacy budget accounting utilities.
The project currently implements a historical daily-budget simulation in
src/experiments/cumulative_privacy.py, which applies basic sequential
additive composition (epsilon_cumulative = sum of per-round epsilons).

Reference:
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy.
    Foundations and Trends in Theoretical Computer Science, 9(3–4), 211–407.
    https://doi.org/10.1561/0400000042
"""
