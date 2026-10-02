import os
import pandas as pd
from typing import List

DETAILED_COLUMNS = [
    "experiment",
    "epsilon",
    "users",
    "events_per_user",
    "total_events",
    "dictionary_size",
    "m",
    "k",
    "repetition",
    "mae",
    "rmse",
    "nmae",
    "top10_overlap",
    "spearman",
    "communication_bytes_per_record",
    "sketch_memory_bytes",
    "sketch_memory_mib"
]

def save_detailed_results_csv(df: pd.DataFrame, filepath: str) -> None:
    """
    Save detailed experiment results DataFrame to CSV with deterministic column ordering.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Ensure all required columns are present
    missing = [c for c in DETAILED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required detailed columns: {missing}")
        
    df_ordered = df[DETAILED_COLUMNS]
    df_ordered.to_csv(filepath, index=False)


def save_summary_results_csv(df: pd.DataFrame, filepath: str, column_order: List[str] = None) -> None:
    """
    Save summary experiment results DataFrame to CSV with deterministic column ordering.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if column_order is not None:
        df_ordered = df[column_order]
    else:
        df_ordered = df
        
    df_ordered.to_csv(filepath, index=False)
