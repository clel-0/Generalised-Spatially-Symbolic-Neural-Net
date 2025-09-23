import numpy as np


def generate_polynomial_dataset(n_samples=1000, max_degree=20, coeff_range=(-5, 5), seed=0):
    """
    Generates random polynomials and their derivatives.
    
    Returns:
        inputs: np.ndarray shape (n_samples, D)
        targets: np.ndarray shape (n_samples, D)
    """
    rng = np.random.default_rng(seed)
    D = max_degree + 1

    inputs = rng.integers(coeff_range[0], coeff_range[1]+1, size=(n_samples, D)).astype(float)
                                
    # Compute derivatives
    powers = np.arange(D - 1, 0, -1)  # [D-1, D-2, ..., 1]
    targets = inputs[:, :-1] * powers  # Drop constant term, scale by degree

    # Pad targets with 0 so input and target shapes match
    targets = np.concatenate([targets, np.zeros((n_samples, 1))], axis=1)

    return inputs, targets