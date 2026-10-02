import urllib.request
import re
import numpy as np
import pandas as pd

EMOJI_URL = "https://www.unicode.org/Public/emoji/latest/emoji-test.txt"

def load_emoji_vocabulary(dictionary_size: int, s: float = 1.07) -> pd.DataFrame:
    """
    Downloads and parses the official Unicode emoji dataset.
    Generates a deterministic vocabulary of fully-qualified emojis.
    
    Args:
        dictionary_size: Number of unique emojis in the vocabulary.
        s: The exponent characterizing the Zipf distribution for simulated probabilities.
        
    Returns:
        pd.DataFrame containing the vocabulary with probabilities.
    """
    if dictionary_size <= 0:
        raise ValueError("Dictionary size must be positive.")

    # Download and parse
    req = urllib.request.Request(EMOJI_URL, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to download emoji dataset: {e}")
        
    emojis = []
    
    # Parse lines
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        # Example valid line:
        # 1F600                                      ; fully-qualified     # 😀 E1.0 grinning face
        if "; fully-qualified" in line:
            # Extract the actual emoji character
            # It usually appears after the '#' sign
            parts = line.split('#')
            if len(parts) >= 2:
                # The text after '#' looks like " 😀 E1.0 grinning face"
                emoji_part = parts[1].strip()
                # The first character/sequence separated by space is the emoji
                emoji_char = emoji_part.split(' ')[0]
                emojis.append(emoji_char)

    # Remove duplicates but preserve some deterministic order
    unique_emojis = []
    seen = set()
    for emj in emojis:
        if emj not in seen:
            seen.add(emj)
            unique_emojis.append(emj)
            
    # Sort deterministically
    unique_emojis.sort()

    if len(unique_emojis) < dictionary_size:
        raise ValueError(f"Requested dictionary size {dictionary_size} exceeds available fully-qualified emojis ({len(unique_emojis)}).")
        
    selected_emojis = unique_emojis[:dictionary_size]
    
    # Ranks 1 to D
    ranks = np.arange(1, dictionary_size + 1)
    
    # Zipf weights: 1 / rank^s
    weights = 1.0 / (ranks ** s)
    
    # Probabilities: weight / sum(weights)
    probabilities = weights / np.sum(weights)
    
    df = pd.DataFrame({
        "item_id": np.arange(dictionary_size),
        "item": selected_emojis,
        "probability": probabilities,
        "rank": ranks,
        "domain": "emoji"
    })
    
    return df
