# W0 = np.array([[1.25, -2.31, 4.52], [-3.20, 0.76, 2.12], [4.87, -1.40, 3.55]])
# b0 = np.array([[1.53], [-0.73], [2.39]])
# W1 = np.array([[2.10, -4.50, 3.00], [1.71, -2.38, 0.85], [3.41, 1.25, -0.94]])
# b1 = np.array([[-1.11], [2.46], [3.60]])
# W2 = np.array([[1.85, -3.25, 2.42], [0.93, 4.17, -2.83]])
# b2 = np.array([[1.21], [-1.79]])

# layer0 = Layer(W0, b0, activation='relu')
# layer1 = Layer(W1, b1, activation='relu')
# layer2 = Layer(W2, b2, activation='id')

# layers = [layer0, layer1, layer2]

# network = Network(layers) 

# intermediate_network = copy.deepcopy(network)

# W0 = np.array([[ 0.99591392, -0.20067469],[-0.43028181, 3.19825846]])
# b0 = np.array([[-1.78736748],[ 1.66302203]])
# W1 = np.array([[-3.50828449, -3.7708556 ]])
# b1 = np.array([[1.99237842]])

# layer0 = Layer(W0, b0, activation='relu')
# layer1 = Layer(W1, b1, activation='id')

# layers = [layer0, layer1]

# network = Network(layers) 

# intermediate_network = copy.deepcopy(network)

structure = [dim, 2, 2]

W0 = np.array([[-1.02946293, -0.76284027],[-3.78575567, -1.04142831]])
b0 = np.array([[-4.28482608], [ 3.38609108]])

W1 = np.array([[2.70114439, 2.46951665]])
b1 = np.array([[4.6516897]])

################################################################################

# n = 1000000
# input_box = np.array([
#         [0.5, 0.5],  
#         [1.5, 1.5],
#         [-1.5, -1.5]
#     ])

# input_box = np.array([[ 2.5798121, 3.5798121],[-0.2398504, 0.7601496]])

################################################################################

# output_box_1 = network.propagate_vertices(input_box)
# print(f"Propagation of vertices through NN:\n {output_box_1}")

# output_box_3 = quantized_network.propagate_vertices(input_box)
# print(f"Propagation of vertices through QNN:\n {output_box_3}")

# out_vectors_NN = network.propagate(vectors)
# print(f"Propagation of vectors through NN:\n {out_vectors_NN}")

# out_vectors_QNN = quantized_network.propagate(vectors)
# print(f"Propagation of vectors through QNN:\n {out_vectors_QNN}")

# output_box_1 = network.propagate_box(input_box)
# print(f"Propagation of box through NN:\n {output_box_1}")

# output_box_2 = network.propagate_vertices(input_box)
# print(f"Propagation of vertices through NN:\n {output_box_2}")

# output_box_3 = quantized_network.propagate_box(input_box)
# print(f"Propagation of box through QNN:\n {output_box_3}")

# output_box_4 = quantized_network.propagate_vertices(input_box)
# print(f"Propagation of vertices through QNN:\n {output_box_4}")

#  def final_concretize_trial(self): ## TO BE WORKED ON
#         """
#         Compute the bounds of the constrained zonotope using the dual formulation.

#         Returns:
#         - A NumPy array of shape (d, 2), where each row is [lower_bound, upper_bound].
#         """

#         d, m = self.W.shape
#         bounds = np.zeros((d, 2))  # To store [lower_bound, upper_bound] for each dimension
#         K = len(self.constraints)

#         A_raw = np.array([constraint[0] for constraint in self.constraints], dtype=object)
#         max_len = max(len(row) for row in A_raw) if A_raw.size > 0 else 0
#         A = np.array([np.pad(row, (0, max_len - len(row)), mode='constant') for row in A_raw], dtype=np.float64)

#         b_vals = np.array([constraint[1] for constraint in self.constraints],  dtype=np.float64)

#         for i in range(d):
#             alpha = self.W[i]  # Coefficients for epsilon
#             beta = self.b[i]  # Center term

#             def objective_lower_bound(lambda_k):
#                 term1 = - np.sum(np.abs(alpha)) + np.sum(np.sum(np.dot(lambda_k, A)))
#                 term2 = - np.sum(lambda_k * b_vals)
#                 return term1 + term2 + beta

#             def objective_upper_bound(lambda_k):
#                 term1 = np.sum(np.abs(alpha)) - np.sum(np.sum(np.dot(lambda_k, A)))
#                 term2 = -np.sum(lambda_k * b_vals)
#                 return term1 + term2 + beta

#             lambda_init_1 = np.zeros(K) if K > 0 else np.zeros(1)
#             lambda_init_2 = np.zeros(K) if K > 0 else np.zeros(1)

#             bounds_lambda = [(0, 2)] * K
            
#             res_1 = minimize(lambda x: -objective_lower_bound(x), lambda_init_1, bounds = bounds_lambda , method='L-BFGS-B')
#             res_2 = minimize(lambda x: -objective_upper_bound(x), lambda_init_2, bounds = bounds_lambda ,  method='L-BFGS-B')
#             lower_bound = - res_1.fun if res_1.success else float('-inf')
#             upper_bound = - res_2.fun if res_2.success else float('+inf')
            
#             bounds[i] = [lower_bound, upper_bound]
        
#         return bounds


################################
# CONSTRAINTS                  #
################################

# if extra_constraint:
#     lambda_cons = u[i]/(u[i]-l[i])
#     mu_cons = - (u[i] * l[i])/(u[i]-l[i])
#     self.constraints.append(((lambda_cons * old_row) - new_row , (lambda_cons * old_bias) + mu_cons - new_bias))

# self.constraints.append(( - new_row, C_ub - new_bias))


################################################################################
################################################################################

import numpy as np
from itertools import combinations
from itertools import product
import copy
from network import Layer 
from network import Network
from network import generate_network
from network import generate_input_box
from network import print_network_parameters
from zonotope import Zonotope
from constrained_zonotope import ConstrainedZonotope
from oracle import sample_vectors_from_box
from abstract import abstract_to_vertices

weight = np.array([[ 0.    ,    0.   ,     0.   ,    -8.038875, 15.12375, 4.356],
 [ 0.  ,      0.   ,     0.  ,     10.314495  , 7.60275 , -5.094]])

bias = np.array([[4.655875],  [21.291445]])

zonotope = Zonotope(weight , bias)

print(zonotope.concretize())