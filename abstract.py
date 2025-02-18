import numpy as np
from itertools import combinations
from itertools import product
from zonotope import Zonotope
from constrained_zonotope import ConstrainedZonotope

def abstract_to_vertices(box):
    """
    Converts a box into a zonotope.

    Parameters:
    - box: A NumPy array of shape (d, 2), where each row represents [lower_bound, upper_bound].

    Returns:
    - vecrtices: vertices of the initial box
    """
    return np.array(list(product(*box)))  

def concretize_to_box(vertices):
    """
    Converts a set of vertices into a bounding box.

    Parameters:
    - vertices: A NumPy array of shape (n, d), where each row is a vertex.

    Returns:
    - box: A NumPy array of shape (d, 2) with [min, max] for each dimension.
    """
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    return np.vstack((min_vals, max_vals)).T
    

def abstract_to_zonotope(box):
    """
    Converts a box into a zonotope.

    Parameters:
    - box: A NumPy array of shape (d, 2), where each row represents [lower_bound, upper_bound].

    Returns:
    - Zonotope: A zonotope representing the abstracted box.
    """
    # Extract the lower and upper bounds
    lower_bounds = box[:, 0]
    upper_bounds = box[:, 1]

    # Compute the center (bias) and the ranges
    center = (lower_bounds + upper_bounds) / 2
    ranges = (upper_bounds - lower_bounds) / 2

    # Create the generator matrix W
    # Each range corresponds to one noise symbol
    W = np.diag(ranges)

    # Return the zonotope
    return Zonotope(W, center)

def abstract_to_constrained_zonotope(box):
    """
    Converts a box into a zonotope.

    Parameters:
    - box: A NumPy array of shape (d, 2), where each row represents [lower_bound, upper_bound].

    Returns:
    - ConstraintedZonotope: A constrainted zonotope representing the abstracted box.
    """
    # Extract the lower and upper bounds
    lower_bounds = box[:, 0]
    upper_bounds = box[:, 1]

    # Compute the center (bias) and the ranges
    center = (lower_bounds + upper_bounds) / 2
    ranges = (upper_bounds - lower_bounds) / 2

    # Create the generator matrix W
    # Each range corresponds to one noise symbol
    W = np.diag(ranges)

    # Return the zonotope
    return ConstrainedZonotope(W, center, [])

def clamp(value, min_val, max_val):
    """
    Clamp a value or array of values to a specified range.

    Parameters:
        value (float or array): The value(s) to clamp.
        min_val (float): The minimum value of the range.
        max_val (float): The maximum value of the range.

    Returns:
        float or array: Clamped value(s).
    """
    return np.minimum(np.maximum(value, min_val), max_val)


