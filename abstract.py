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

def relu_interval_propagation(intervals):
    """
    Propagates intervals through a ReLU activation function.
    
    Parameters:
    - intervals: A NumPy array of shape (n, 2), where each row is [l, u].
    
    Returns:
    - A NumPy array of shape (n, 2) with propagated intervals.
    """
    propagated_intervals = np.zeros_like(intervals)

    l, u = intervals[:, 0], intervals[:, 1]  # Extract lower and upper bounds
    
    # Fully negative case: Map to [0, 0]
    propagated_intervals[:, 0] = np.maximum(l, 0)  # Lower bound
    propagated_intervals[:, 1] = np.maximum(u, 0)  # Upper bound

    return propagated_intervals

def clamp_interval_propagation(intervals, lower_bound, upper_bound):
        """
        Clamps the propagated interval values within a specified range.
        
        Parameters:
        - intervals: NumPy array of shape (n, 2), where each row is [l, u].
        - lower_bound: Float or NumPy array of shape (n,) representing the minimum allowed values.
        - upper_bound: Float or NumPy array of shape (n,) representing the maximum allowed values.

        Returns:
        - A NumPy array of shape (n, 2) representing the clamped intervals.
        """
        clamped_lower = np.maximum(intervals[:, 0], lower_bound)
        clamped_upper = np.minimum(intervals[:, 1], upper_bound)
        return np.vstack((clamped_lower, clamped_upper)).T

def linear_interval_propagation(intervals, W, b):
    """
    Propagates intervals through a linear transformation.

    Parameters:
    - W: NumPy array of shape (m, n), the weight matrix.
    - b: NumPy array of shape (m,), the bias vector.
    - intervals: NumPy array of shape (n, 2), where each row is [l, u].

    Returns:
    - A NumPy array of shape (m, 2) representing the transformed intervals.
    """

    l, u = intervals[:, 0], intervals[:, 1]

    # Compute transformed lower and upper bounds considering sign of W
    lower_bounds = W @ l[:, np.newaxis] + b 
    upper_bounds = W @ u[:, np.newaxis] + b

    return np.vstack((lower_bounds.T[0], upper_bounds.T[0])).T

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


