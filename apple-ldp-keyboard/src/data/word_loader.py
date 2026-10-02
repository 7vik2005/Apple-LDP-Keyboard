import numpy as np
import pandas as pd

def generate_word_vocabulary(dictionary_size: int, s: float = 1.07) -> pd.DataFrame:
    """
    Generates a deterministic synthetic word vocabulary with a Zipf-like distribution.
    
    Args:
        dictionary_size: Number of unique words in the vocabulary.
        s: The exponent characterizing the Zipf distribution.
        
    Returns:
        pd.DataFrame containing the vocabulary with probabilities.
    """
    if dictionary_size <= 0:
        raise ValueError("Dictionary size must be positive.")

    # Generate synthetic items: word_0, word_1, ...
    items = [f"word_{i}" for i in range(dictionary_size)]
    
    # Ranks 1 to D
    ranks = np.arange(1, dictionary_size + 1)
    
    # Zipf weights: 1 / rank^s
    weights = 1.0 / (ranks ** s)
    
    # Probabilities: weight / sum(weights)
    probabilities = weights / np.sum(weights)
    
    df = pd.DataFrame({
        "item_id": np.arange(dictionary_size),
        "item": items,
        "probability": probabilities,
        "rank": ranks,
        "domain": "word"
    })
    
    return df
