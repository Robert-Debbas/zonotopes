import numpy as np
from itertools import combinations
from itertools import product
from scipy.optimize import minimize


class ConstrainedZonotope:
    def __init__(self, W, b, constraints):
        """
        Initialize a constrained zonotope.

        Parameters:
        - W: numpy array of shape (d, m), the generator matrix.
        - b: numpy array of shape (d,), the center vector.
        - constraints: list of tuples (A_k, b_k) representing the linear constraints.
                       Each A_k is a numpy array of shape (1, m), and each b_k is a float.
        """
        self.W = np.array(W)  # Ensure W is a numpy array
        self.b = np.array(b)  # Ensure b is a numpy array
        self.constraints = constraints  # List of (A_k, b_k)

        # Validate dimensions
        if self.W.ndim != 2 or self.b.ndim != 1:
            raise ValueError("W must be a 2D array and b must be a 1D array.")
        if self.W.shape[0] != self.b.shape[0]:
            raise ValueError("The number of rows in W must match the size of b.")

    def linear_transformation(weights, biases):
        
        self.W = np.dot(weights, self.W) 
        self.b = np.dot(weights, self.b) + biases 
        return self
    
    def concretize(self): ## TO BE WORKED ON
        """
        Compute the bounds of the constrained zonotope using the dual formulation.

        Returns:
        - A NumPy array of shape (d, 2), where each row is [lower_bound, upper_bound].
        """

        d, m = self.W.shape
        bounds = np.zeros((d, 2))  # To store [lower_bound, upper_bound] for each dimension

        for i in range(d):
            alpha = self.W[i]  # Coefficients for epsilon
            beta = self.b[i]  # Center term

            A = np.array([constraint[0] for constraint in self.constraints])
            b_vals = np.array([constraint[1] for constraint in self.constraints])
            K = len(self.constraints)
            print(self.constraints)

            def objective_lower_bound(lambda_k):
                term1 = -np.sum(np.abs(alpha) + np.sum(lambda_k[:, None] * A, axis=0))
                term2 = -np.sum(lambda_k * b_vals)
                return term1 + term2 + beta

            def objective_upper_bound(lambda_k):
                term1 = np.sum(np.abs(alpha) - np.sum(lambda_k[:, None] * A, axis=0))
                term2 = -np.sum(lambda_k * b_vals)
                return term1 + term2 + beta

            lambda_init_1 = np.zeros(K)
            lambda_init_2 = np.zeros(K)

            bounds_lambda = [(0, None)] * K if K > 0 else None
            
            res_1 = minimize(lambda x: -objective_lower_bound(x), lambda_init_1,bounds , method='L-BFGS-B')
            res_2 = minimize(lambda x: -objective_upper_bound(x), lambda_init_2, method='L-BFGS-B')
            lower_bound = - res_1.fun if res_1.success else float('-inf')
            upper_bound = - res_2.fun if res_2.success else float('+inf')
            
            bounds[i] = [lower_bound, upper_bound]
        
        return bounds


    def abstract_ReLU(self):
        """
        Abstracts the ReLU function over the current constrainted zonotope.

        Returns:
        - ConstraintedZonotope: A new ConstraintedZonotope object representing the abstracted ReLU.
        """

        bounds = self.concretize() 
        constraints = self.constraints

        W_new = []
        b_new = []

        for i in range(self.W.shape[0]):

            l, u = bounds[i][0], bounds[i][1]

            if u <= 0:
                # Fully negative case: y = 0
                a_param = 0
                b_param = 0
                
            elif l >= 0:
                # Fully positive case: y = x
                a_param = 1
                b_param = 0

            else:
                # Unstable case: l < 0 < u
                a_param = u / (u - l)
                b_param = -u * l / (u - l)
            

            old_row =  np.hstack((self.W[i], 0))
            old_bias = self.b[i]

            new_row = np.hstack((a_param * self.W[i], b_param / 2))
            new_bias = a_param * self.b[i] + b_param / 2

            W_new.append(new_row)
            b_new.append(new_bias)

            constraints.append((np.array(new_row), new_bias))
            constraints.append((np.array(new_row - old_row), new_bias - old_bias))

        return ConstrainedZonotope(W_new, b_new, constraints)

    def abstract_clamp(self, C_ub):
        """
        Abstracts the clamp(x, 0, C^{ub}) function over the current zonotope.

        Parameters:
        - C_ub: float, the upper bound of the clamp function.

        Returns:
        - Zonotope: A new Zonotope object representing the abstracted clamp.
        """
        bounds = self.concretize()  
        constraints = self.constraints

        W_new = []
        b_new = []

        for i in range(self.W.shape[0]):

            l, u = bounds[i][0], bounds[i][1]

            if u <= 0:
                # Fully negative case: clamp(x, 0, C^{ub}) = 0
                new_row = np.hstack((np.zeros(self.W.shape[1]), 0)) # Append 0 for the new noise dimension
                new_bias = 0

            elif l >= C_ub:
                # Fully above C^{ub}: clamp(x, 0, C^{ub}) = C^{ub}
                new_row = np.hstack((np.zeros(self.W.shape[1]), 0))  # No noise
                new_bias = C_ub

            elif l >= 0 and u <= C_ub:
                # Fully within the range [0, C^{ub}]: clamp(x, 0, C^{ub}) = x
                new_row = np.hstack((self.W[i], 0)) # Append 0 for the new noise dimension
                new_bias = self.b[i]

            else:

                # Mixed case: clamp(x, 0, C^{ub}) needs abstraction
                if l < 0 and C_ub <= u:
                    # Case: l <= 0 <= C^{ub} <= u
                    a_param = np.min((C_ub / (C_ub - l)), (C_ub / u))
                    b_param = np.max((1 - a_param) * C_ub, - a_param * l)

                    # Adjust W and add new noise symbol
                    new_row = np.hstack((a_param * self.W[i], b_param / 2))
                    new_bias = a_param * self.b[i] + b_param / 2

                elif l < 0 and u <= C_ub:
                    # Case: l <= 0 < u <= C^{ub}
                    a_param = u / (u - l)
                    b_param = -u * l / (u - l)

                    new_row = np.hstack((a_param * self.W[i], b_param / 2))
                    new_bias = a_param * self.b[i] + b_param / 2


                elif l >= 0 and C_ub < u:
                    # Case: 0 <= l <= C^{ub} < u
                    a_param = (C_ub - l) / (u - l)
                    b_param = (C_ub - l) * (1 - a_param)

                    new_row = np.hstack((a_param * self.W[i], b_param / 2))
                    new_bias = a_param * self.b[i] + b_param / 2

            old_row =  np.hstack((self.W[i], 0))
            old_bias = self.b[i]

            W_new.append(new_row)
            b_new.append(new_bias)

            constraints.append((np.array(new_row), new_bias))
            constraints.append((np.array(new_row - old_row), new_bias - old_bias))
            constraints.append(( - np.array(new_row), C_ub - new_bias))

        W_new = np.array(W_new)
        b_new = np.array(b_new)

        return ConstrainedZonotope(W_new, b_new, constraints)

    def abstract_round(self):
        """
        Abstracts the rounding function over the current zonotope.

        Parameters:
        - lambda_val: Regularization parameter for the least squares computation.

        Returns:
        - Zonotope: A new Zonotope object representing the abstracted round function.
        """
        W_new = []
        # Iterate over each row of the constrained zonotope
        for i in range(len(self.b)):
            new_row = np.hstack((self.W[i], 0.5))
            W_new.append(new_row)
        W_new = np.array(W_new)
        # Return the new constrained zonotope
        return ConstrainedZonotope(W_new, self.b, self.constraints)
