import pytest
import numpy as np
from src.data.word_loader import generate_word_vocabulary
from src.data.emoji_loader import load_emoji_vocabulary
from src.data.dataset_generator import generate_users_and_events, compute_ground_truth

def test_zipf_probabilities_sum_to_one():
    df = generate_word_vocabulary(50)
    assert np.isclose(df['probability'].sum(), 1.0)

def test_vocabulary_ids_are_unique():
    df = generate_word_vocabulary(50)
    assert df['item_id'].is_unique

def test_vocabulary_size_is_correct():
    df = generate_word_vocabulary(123)
    assert len(df) == 123

def test_event_count():
    vocab_df = generate_word_vocabulary(50)
    events_df = generate_users_and_events(num_users=10, events_per_user=20, dictionary_df=vocab_df, seed=42, domain="word")
    assert len(events_df) == 200

def test_every_event_item_in_vocabulary():
    vocab_df = generate_word_vocabulary(50)
    events_df = generate_users_and_events(10, 20, vocab_df, 42, "word")
    valid_items = set(vocab_df['item_id'])
    assert events_df['item_id'].isin(valid_items).all()

def test_ground_truth_count_sum():
    vocab_df = generate_word_vocabulary(50)
    events_df = generate_users_and_events(10, 20, vocab_df, 42, "word")
    gt_df = compute_ground_truth(events_df, vocab_df)
    assert gt_df['true_count'].sum() == 200

def test_same_seed_identical_vocabulary():
    df1 = generate_word_vocabulary(50)
    df2 = generate_word_vocabulary(50)
    assert df1.equals(df2)

def test_same_seed_identical_events():
    vocab_df = generate_word_vocabulary(50)
    events_df1 = generate_users_and_events(10, 20, vocab_df, 42, "word")
    events_df2 = generate_users_and_events(10, 20, vocab_df, 42, "word")
    assert events_df1.equals(events_df2)

def test_different_seed_different_events():
    vocab_df = generate_word_vocabulary(50)
    events_df1 = generate_users_and_events(10, 20, vocab_df, 42, "word")
    events_df2 = generate_users_and_events(10, 20, vocab_df, 99, "word")
    assert not events_df1.equals(events_df2)

def test_emoji_parser_unique():
    # Only pull a small amount to avoid too much overhead in tests if possible, 
    # but since it's downloading the whole file, we test with a small subset.
    df = load_emoji_vocabulary(10)
    assert df['item'].is_unique
    # Should all be fully-qualified according to parser logic
    assert len(df) == 10
