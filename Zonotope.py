import numpy as np
from itertools import combinations
from itertools import product

class Zonotope:
    def __init__(self, W, b):
        """
        Initialize a zonotope.

        Parameters:
        - W: numpy array of shape (d, m), the generator matrix.
        - b: numpy array of shape (d,), the center vector.
        """
        self.W = np.array(W)  # Ensure W is a numpy array
        self.b = np.array(b)  # Ensure b is a numpy array
        
        # Validate dimensions
        if self.W.ndim != 2 or self.b.ndim != 1:
            raise ValueError("W must be a 2D array and b must be a 1D array.")
        if self.W.shape[0] != self.b.shape[0]:
            raise ValueError("The number of rows in W must match the size of b.")
    
    def linear_transformation(self, weights, biases):

        self.W = np.dot(weights, self.W) 

        self.b = np.dot(weights, self.b) + biases.T
        self.b = self.b.T

        return self

    def bounds(self):
        """
        Compute the bounds of the zonotope and return as an array of intervals.

        Returns:
        - A NumPy array of shape (d, 2), where each row is [lower_bound, upper_bound].
        """

        abs_W = np.abs(self.W)

        lower_bounds = self.b.T - np.sum(abs_W, axis=1)
        lower_bounds = lower_bounds.T

        upper_bounds = self.b.T + np.sum(abs_W, axis=1)
        upper_bounds = upper_bounds.T

        return lower_bounds, upper_bounds
    
    def concretize(self):
        """
        Compute the bounds of the zonotope and return as an array of intervals.

        Returns:
        - A NumPy array of shape (d, 2), where each row is [lower_bound, upper_bound].
        """
         
        abs_W = np.abs(self.W)

        lower_bounds = self.b.T - np.sum(abs_W, axis=1)
        upper_bounds = self.b.T + np.sum(abs_W, axis=1)

        intervals = np.stack((lower_bounds, upper_bounds), axis = 1)

        return intervals[0].T

    def abstract_ReLU(self):
        """
        Abstracts the ReLU function over the current zonotope.

        Returns:
        - Zonotope: A new Zonotope object representing the abstracted ReLU.
        """
        l, u = self.bounds() 

        W_new = np.zeros((len(self.W), len(self.W[0])+1))
        b_new = np.zeros((len(self.b)))

        for i in range(self.W.shape[0]):

            if u[i] <= 0:
                # Fully negative case: y = 0
                W_new[i] = np.hstack((np.zeros(self.W.shape[1]), 0))  # Append 0 for the new noise dimension
                b_new[i] = 0
            elif l[i] >= 0:
                # Fully positive case: y = x
                W_new[i] = np.hstack((self.W[i], 0)) # Append 0 for the new noise dimension
                b_new[i] = self.b[i]
            else:
                # Unstable case: l[i] < 0 < u[i]
                a_param = u[i] / (u[i] - l[i])
                b_param = -u[i] * l[i] / (u[i] - l[i])

                # Adjust W and add new noise symbol
                W_new[i] = np.hstack((a_param * self.W[i], b_param/2))
                b_new[i] = a_param * self.b[i] + b_param/2

        self.W = np.array(W_new)
        self.b = np.array(b_new)

        return self

    def abstract_clamp(self, C_ub):
        """
        Abstracts the clamp(x, 0, C^{ub}) function over the current zonotope.

        Parameters:
        - C_ub: float, the upper bound of the clamp function.

        Returns:
        - Zonotope: A new Zonotope object representing the abstracted clamp.
        """
        l, u = self.bounds()  # Get the bounds of the zonotope

        W_new = np.zeros((len(self.W), len(self.W[0]) + 1))
        b_new = np.zeros((len(self.b)))

        for i in range(self.W.shape[0]):

            if u[i] <= 0:
                # Fully negative case: clamp(x, 0, C^{ub}) = 0
                W_new[i] = np.hstack((np.zeros(self.W.shape[1]), 0))  # Append 0 for the new noise dimension
                b_new[i] = 0

            elif l[i] >= C_ub:
                # Fully above C^{ub}: clamp(x, 0, C^{ub}) = C^{ub}
                W_new[i] = np.hstack((np.zeros(self.W.shape[1]), 0))  # No noise
                b_new[i] = C_ub

            elif l[i] >= 0 and u[i] <= C_ub:
                # Fully within the range [0, C^{ub}]: clamp(x, 0, C^{ub}) = x
                W_new[i] = np.hstack((self.W[i], 0)) # Append 0 for the new noise dimension
                b_new[i] = self.b[i]

            else:

                # Mixed case: clamp(x, 0, C^{ub}) needs abstraction
                if l[i] < 0 and C_ub <= u[i]:
                    # Case: l <= 0 <= C^{ub} <= u
                    a_param = min((C_ub / (C_ub - l[i])), (C_ub / u[i]))
                    b_param = max((1 - a_param) * C_ub, - a_param * l[i])

                elif l[i] < 0 and u[i] <= C_ub:
                    # Case: l <= 0 < u <= C^{ub}
                    a_param = u[i] / (u[i] - l[i])
                    b_param = - u[i] * l[i] / (u[i] - l[i])

                elif l[i] >= 0 and C_ub < u[i]:
                    # Case: 0 <= l <= C^{ub} < u
                    a_param = (C_ub - l[i]) / (u[i] - l[i])
                    b_param = (C_ub - l[i]) * (1 - a_param)

                W_new[i] = np.hstack((a_param * self.W[i], b_param / 2))
                b_new[i] = a_param * self.b[i] + b_param / 2

        self.W = W_new
        self.b = b_new

        return self

    def abstract_round(self): 
        """
        Abstracts the rounding function over the current zonotope.

        Parameters:
        - lambda_val: Regularization parameter for the least squares computation.

        Returns:
        - Zonotope: A new Zonotope object representing the abstracted round function.
        """

        W_new = np.zeros((len(self.W), len(self.W[0]) + 1))
        for i in range(len(self.b)): W_new[i] = np.hstack((self.W[i], 0.5))
        self.W = W_new
        return self