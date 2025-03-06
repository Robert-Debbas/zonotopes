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

    def linear_transformation(self, weights, biases):
        
        self.W = np.dot(weights, self.W) 

        self.b = np.dot(weights, self.b) + biases.T
        self.b = self.b.T

        return self


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

    def final_concretize(self):
        """
        Compute the bounds of the constrained zonotope using the dual formulation.

        Returns:
        - A NumPy array of shape (d, 2), where each row is [lower_bound, upper_bound].
        """

        d, m = self.W.shape
        bounds = np.zeros((d, 2))
        K = len(self.constraints)

        A_raw = np.array([constraint[0] for constraint in self.constraints], dtype=object)
        max_len = max(len(row) for row in A_raw) if A_raw.size > 0 else 0
        A = np.array([np.pad(row, (0, max(m, max_len) - len(row)), mode='constant') for row in A_raw], dtype=np.float64)

        b_vals = np.array([constraint[1] for constraint in self.constraints],  dtype=np.float64)

        constraints = []

        for j in range(K): constraints.append({'type': 'ineq', 'fun': lambda x: A[j] @ x.T + b_vals[j]})

        for i in range(d):

            alpha = self.W[i]  # Coefficients for epsilon
            beta = self.b[i]  # Center term

            def objective_function(eps): return np.dot(alpha, eps) + beta

            eps_init_1 = np.zeros(m) 
            eps_init_2 = np.zeros(m) 

            bounds_eps = [(-1, 1)] * m

            
            res_1 = minimize(lambda x : objective_function(x), eps_init_1, bounds = bounds_eps, constraints = constraints, method='SLSQP')
            res_2 = minimize(lambda x :  - objective_function(x) , eps_init_2, bounds = bounds_eps, constraints = constraints, method='SLSQP')

            lower_bound = res_1.fun if res_1.success else float('-inf')
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

        l, u = bounds.T[0], bounds.T[1]

        W_new = np.zeros((len(self.W), len(self.W[0])+1))
        b_new = np.zeros(len(self.b))

        for i in range(self.W.shape[0]):

            if u[i] <= 0:
                # Fully negative case: y = 0
                new_row = np.hstack((np.zeros(len(self.W[0])), 0))  # Append 0 for the new noise dimension
                new_bias = 0

            elif l[i] >= 0:
                # Fully positive case: y = x
                new_row = np.hstack((self.W[i], 0)) # Append 0 for the new noise dimension
                new_bias = self.b[i]

            else:

                # Unstable case: l[i] < 0 < u[i]
                a_param = u[i] / (u[i] - l[i])
                b_param = - u[i] * l[i] / (u[i] - l[i])

                # Adjust W and add new noise symbol
                new_row = np.hstack((a_param * self.W[i], b_param/2))
                new_bias = a_param * self.b[i] + b_param/2

                old_row = np.hstack((self.W[i], 0))
                old_bias = self.b[i]

                self.constraints.append((new_row, new_bias))
                self.constraints.append((new_row - old_row, new_bias - old_bias))
                
            W_new[i] = new_row
            b_new[i] = new_bias

        self.W = W_new 
        self.b = b_new 

        return self

    def abstract_ReLU_new(self):
        """
        Abstracts the ReLU function over the current constrainted zonotope.

        Returns:
        - ConstraintedZonotope: A new ConstraintedZonotope object representing the abstracted ReLU.
        """

        bounds = self.concretize() 

        l, u = bounds.T[0], bounds.T[1]

        extension = 0

        delta = 0

        W_new = []
        b_new = []

        for i in range(self.W.shape[0]):

            if u[i] <= 0:
                # Fully negative case: y = 0
                new_row = np.zeros(len(self.W[0]))  # Append 0 for the new noise dimension
                new_bias = 0

            elif l[i] >= 0:
                # Fully positive case: y = x
                new_row = self.W[i] # Append 0 for the new noise dimension
                new_bias = self.b[i][0]

            else:

                extension += 1

                b_param = u[i] 

                # Adjust W and add new noise symbol
                new_row = np.concatenate((np.zeros(len(self.W[0]) + extension - 1), np.array([b_param/2])))
                new_bias = b_param/2

                old_row = np.concatenate((self.W[i], np.zeros(extension)))
                old_bias = self.b[i][0]

                lambda_cons_2 = u[i] / (u[i] - l[i])
                mu_cons_2 = - (u[i] * l[i]) / (u[i] - l[i])

                constraint_1 = (new_row - old_row, new_bias - old_bias)
                constraint_2 = (lambda_cons_2 * old_row - new_row, lambda_cons_2 * old_bias + mu_cons_2 - new_bias)

                self.constraints.append(constraint_1)
                self.constraints.append(constraint_2)
                
            W_new.append(new_row)
            b_new.append(new_bias)

        max_cols = max(arr.shape[0] for arr in W_new) 
    
        padded_matrix = np.zeros((len(W_new), max_cols))
        
        for i, arr in enumerate(W_new): padded_matrix[i, :arr.shape[0]] = arr

        W_new = padded_matrix
        b_new = np.array(b_new)

        self.W = W_new 
        self.b = b_new 

        return self

    def abstract_clamp(self, C_ub):
        """
        Abstracts the clamp(x, 0, C^{ub}) function over the current zonotope.

        Parameters:
        - C_ub: float, the upper bound of the clamp function.

        Returns:
        - Zonotope: A new Zonotope object representing the abstracted clamp.
        """

        bounds = self.concretize() 

        l, u = bounds.T[0], bounds.T[1]

        W_new = np.zeros((len(self.W), len(self.W[0]) + 1))
        b_new = np.zeros((len(self.b)))

        for i in range(self.W.shape[0]):

            if u[i] <= 0:
                # Fully negative case: clamp(x, 0, C^{ub}) = 0
                new_row = np.hstack((np.zeros(self.W.shape[1]), 0))  # Append 0 for the new noise dimension
                new_bias = 0

            elif l[i] >= C_ub:
                # Fully above C^{ub}: clamp(x, 0, C^{ub}) = C^{ub}
                new_row = np.hstack((np.zeros(self.W.shape[1]), 0))  # No noise
                new_bias = C_ub

            elif l[i] >= 0 and u[i] <= C_ub:
                # Fully within the range [0, C^{ub}]: clamp(x, 0, C^{ub}) = x
                new_row = np.hstack((self.W[i], 0)) # Append 0 for the new noise dimension
                new_bias = self.b[i]

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

                old_row = np.hstack((self.W[i], 0))
                old_bias = self.b[i]

                new_row = np.hstack((a_param * self.W[i], b_param / 2))
                new_bias = a_param * self.b[i] + b_param / 2

                self.constraints.append((new_row,  new_bias))
                self.constraints.append((new_row - old_row, new_bias - old_bias))

            W_new[i] = new_row
            b_new[i] = new_bias

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
