import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

from .word_loader import generate_word_vocabulary
from .emoji_loader import load_emoji_vocabulary

def generate_users_and_events(num_users: int, events_per_user: int, dictionary_df: pd.DataFrame, seed: int, domain: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    
    # Generate user IDs
    user_ids = [f"user_{i:06d}" for i in range(1, num_users + 1)]
    
    total_events = num_users * events_per_user
    
    # Pre-allocate for performance
    event_user_ids = np.repeat(user_ids, events_per_user)
    event_ids = [f"event_{i:06d}" for i in range(1, total_events + 1)]
    
    # Sample items using dictionary probabilities
    item_ids = dictionary_df['item_id'].values
    probabilities = dictionary_df['probability'].values
    
    sampled_item_ids = rng.choice(item_ids, size=total_events, p=probabilities)
    
    # Map item_ids to items
    item_map = dict(zip(dictionary_df['item_id'], dictionary_df['item']))
    sampled_items = [item_map[i] for i in sampled_item_ids]
    
    events_df = pd.DataFrame({
        "user_id": event_user_ids,
        "event_id": event_ids,
        "item_id": sampled_item_ids,
        "item": sampled_items,
        "domain": domain
    })
    
    return events_df

def compute_ground_truth(events_df: pd.DataFrame, dictionary_df: pd.DataFrame) -> pd.DataFrame:
    total_events = len(events_df)
    
    # Count occurrences
    counts = events_df['item_id'].value_counts().reset_index()
    counts.columns = ['item_id', 'true_count']
    
    # Merge with dictionary to ensure all items are present (even those with 0 counts)
    gt_df = pd.merge(dictionary_df[['item_id', 'item', 'domain']], counts, on='item_id', how='left')
    gt_df['true_count'] = gt_df['true_count'].fillna(0).astype(int)
    
    # Compute true probabilities
    gt_df['true_probability'] = gt_df['true_count'] / total_events
    
    # Sort by true count descending, then by item_id
    gt_df = gt_df.sort_values(by=['true_count', 'item_id'], ascending=[False, True]).reset_index(drop=True)
    
    return gt_df

def validate_vocabulary(df: pd.DataFrame, expected_size: int):
    assert len(df) == expected_size, f"Vocab size {len(df)} != {expected_size}"
    assert df['item_id'].is_unique, "item_ids are not unique"
    assert df['item'].is_unique, "items are not unique"
    assert (df['probability'] > 0).all(), "Found non-positive probabilities"
    assert np.isclose(df['probability'].sum(), 1.0), f"Probabilities sum to {df['probability'].sum()}"
    assert df['rank'].is_unique, "ranks are not unique"
    assert (df['rank'].values == np.arange(1, expected_size + 1)).all(), "ranks are not consecutive starting from 1"
    
def validate_events(df: pd.DataFrame, num_users: int, events_per_user: int, vocab_df: pd.DataFrame, expected_domain: str):
    expected_total = num_users * events_per_user
    assert len(df) == expected_total, f"Events count {len(df)} != {expected_total}"
    assert df['event_id'].is_unique, "event_ids are not unique"
    
    valid_items = set(vocab_df['item_id'])
    assert df['item_id'].isin(valid_items).all(), "Found item_ids not in vocabulary"
    assert (df['domain'] == expected_domain).all(), f"Domain mismatch. Expected {expected_domain}"

def validate_ground_truth(df: pd.DataFrame, total_events: int, vocab_size: int):
    assert len(df) == vocab_size, f"Ground truth rows {len(df)} != {vocab_size}"
    assert df['true_count'].sum() == total_events, f"Sum of true_counts {df['true_count'].sum()} != {total_events}"
    assert np.isclose(df['true_probability'].sum(), 1.0), f"Sum of true_probabilities {df['true_probability'].sum()}"
    assert not df['true_count'].isna().any(), "Found NaN in true_counts"

def print_summary(domain, num_users, events_per_user, total_events, dict_size, seed, gt_df):
    print("Dataset Summary")
    print("---------------")
    print(f"Domain: {domain}")
    print(f"Users: {num_users}")
    print(f"Events per user: {events_per_user}")
    print(f"Total events: {total_events}")
    print(f"Dictionary size: {dict_size}")
    print(f"Seed: {seed}")
    print()
    print("Most frequent items:")
    for idx, row in gt_df.head(10).iterrows():
        print(f"{idx+1}. {row['item']} (count: {row['true_count']}, prob: {row['true_probability']:.4f})")
    print()

def main():
    parser = argparse.ArgumentParser(description="Generate dataset for Apple LDP Keyboard.")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Fixed parameters for phase 2
    seed = config.get("seed", 42)
    dict_size = config.get("dictionary_size")
    # For Phase 2, if config is apple_like, it doesn't specify num_users, but the prompt says:
    # "The implementation must also support: configs/apple_like.json with: dictionary_size = 2600... Only create the data required for those future phases."
    # Let's default num_users to 1000 if not specified
    num_users = config.get("num_users", 1000)
    events_per_user = 20

    is_apple = "apple_like" in args.config.lower()
    domain = "emoji" if is_apple else "word"
    
    print("Generating vocabulary...")
    if domain == "word":
        vocab_df = generate_word_vocabulary(dict_size)
        vocab_path = "data/generated/keyboard_vocabulary.csv"
    else:
        vocab_df = load_emoji_vocabulary(dict_size)
        vocab_path = "data/generated/emoji_vocabulary.csv"
        
    vocab_df.to_csv(vocab_path, index=False)
    
    print("Validating vocabulary...")
    validate_vocabulary(vocab_df, dict_size)
    
    print("Generating events...")
    events_df = generate_users_and_events(num_users, events_per_user, vocab_df, seed, domain)
    events_path = "data/generated/keyboard_events.csv"
    events_df.to_csv(events_path, index=False)
    
    print("Validating events...")
    validate_events(events_df, num_users, events_per_user, vocab_df, domain)
    
    print("Computing ground truth...")
    gt_df = compute_ground_truth(events_df, vocab_df)
    gt_path = "data/generated/ground_truth_frequencies.csv"
    gt_df.to_csv(gt_path, index=False)
    
    print("Validating ground truth...")
    validate_ground_truth(gt_df, num_users * events_per_user, dict_size)
    
    print("Validation status: ALL PASS\n")
    print(f"Output files saved to data/generated/")
    print(f" - {vocab_path}")
    print(f" - {events_path}")
    print(f" - {gt_path}\n")
    
    print_summary(domain, num_users, events_per_user, num_users * events_per_user, dict_size, seed, gt_df)

if __name__ == "__main__":
    main()
