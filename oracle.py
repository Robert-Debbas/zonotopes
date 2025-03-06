import numpy as np

def sample_vectors_from_box(box, n):
    """
    Generates a list of n vectors within a given box.

    Parameters:
    - box: NumPy array of shape (d, 2), where each row is [lower_bound, upper_bound].
    - n: Integer, number of vectors to generate.

    Returns:
    - A NumPy array of shape (n, d) containing the generated vectors.
    """
    lower_bounds = box[:, 0]  # Extract lower bounds
    upper_bounds = box[:, 1]  # Extract upper bounds
    
    # Generate n random samples in the given box
    samples = np.random.uniform(lower_bounds, upper_bounds, size=(n, len(lower_bounds)))
    
    return samples
