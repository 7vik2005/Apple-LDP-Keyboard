import numpy as np

def randomization_probability(epsilon: float) -> float:
    """
    Calculate the response probability p for CMS sign-vector randomization under epsilon-LDP:
    p_CMS = exp(epsilon / 2) / (exp(epsilon / 2) + 1) = 1 / (1 + exp(-epsilon / 2))
    
    Parameters
    ----------
    epsilon : float
        Privacy parameter (epsilon > 0).
        
    Returns
    -------
    float
        Probability p in (0.5, 1.0).
    """
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive (epsilon > 0), got {epsilon}")
    
    # Numerically stable calculation of 1 / (1 + exp(-epsilon / 2))
    return 1.0 / (1.0 + np.exp(-epsilon / 2.0))


def hcms_randomization_probability(epsilon: float) -> float:
    """
    Calculate the response probability p for HCMS single-bit randomization under epsilon-LDP:
    p_HCMS = exp(epsilon) / (exp(epsilon) + 1) = 1 / (1 + exp(-epsilon))
    
    Parameters
    ----------
    epsilon : float
        Privacy parameter (epsilon > 0).
        
    Returns
    -------
    float
        Probability p in (0.5, 1.0).
    """
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive (epsilon > 0), got {epsilon}")
        
    # Numerically stable calculation of 1 / (1 + exp(-epsilon))
    return 1.0 / (1.0 + np.exp(-epsilon))


def randomize_sign_vector(vector: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """
    Independently randomize each component of a {-1, +1} sign vector under epsilon-LDP for CMS.
    
    For each position l:
        b_l = +1 with probability p = exp(epsilon/2) / (exp(epsilon/2) + 1)
        b_l = -1 with probability 1 - p
        v_tilde_l = v_l * b_l
        
    Parameters
    ----------
    vector : np.ndarray
        Input vector containing values in {-1, +1}.
    epsilon : float
        Privacy parameter (epsilon > 0).
    rng : np.random.Generator
        Explicit NumPy random generator.
        
    Returns
    -------
    np.ndarray
        Privatized vector v_tilde in {-1, +1}^m.
    """
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive (epsilon > 0), got {epsilon}")
    
    if rng is None or not isinstance(rng, np.random.Generator):
        raise ValueError("A valid np.random.Generator instance (rng) must be provided.")
    
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
        
    if not np.all((vector == 1) | (vector == -1)):
        raise ValueError("Input vector values must be strictly -1 or +1.")
        
    p = randomization_probability(epsilon)
    
    # Sample b_l in {-1, +1}: b_l = +1 if U < p else -1
    u = rng.random(size=vector.shape)
    b = np.where(u < p, 1, -1)
    
    return vector * b


def randomize_bit(bit: int, epsilon: float, rng: np.random.Generator) -> int:
    """
    Randomize a single scalar bit in {-1, +1} under epsilon-LDP for HCMS (Algorithm 6).
    
    b = +1 with probability p_HCMS = exp(epsilon) / (exp(epsilon) + 1)
    b = -1 with probability 1 - p_HCMS
    w_tilde = bit * b
    
    Parameters
    ----------
    bit : int
        Input scalar bit in {-1, +1}.
    epsilon : float
        Privacy parameter (epsilon > 0).
    rng : np.random.Generator
        Explicit NumPy random generator.
        
    Returns
    -------
    int
        Privatized scalar bit w_tilde in {-1, +1}.
    """
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive (epsilon > 0), got {epsilon}")
    if rng is None or not isinstance(rng, np.random.Generator):
        raise ValueError("A valid np.random.Generator instance (rng) must be provided.")
    if bit not in (-1, 1):
        raise ValueError(f"Input bit must be strictly -1 or +1, got {bit}")
        
    p = hcms_randomization_probability(epsilon)
    u = rng.random()
    b = 1 if u < p else -1
    
    return int(bit * b)
