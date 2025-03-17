import numpy as np
from itertools import combinations
from itertools import product
import copy
from network import Layer 
from network import Network
from network import generate_network
from network import generate_input_box
from network import print_network_parameters
from network import quantization_error
from network import abstraction_error
from zonotope import Zonotope
from constrained_zonotope import ConstrainedZonotope
from oracle import sample_vectors_from_box
from abstract import abstract_to_vertices


################################################################
# CONSTRUCT NETWORKS                                           #             
################################################################

dim = 3

structure = [dim, 3, 1]

network = generate_network(structure)

# W0 = np.array([[-1.02946293, -0.76284027],[-3.78575567, -1.04142831]])
# b0 = np.array([[-4.28482608], [ 3.38609108]])
# W1 = np.array([[2.70114439, 2.46951665]])
# b1 = np.array([[4.6516897]])

# W0 = np.array([[1.25, -2.31, 4.52], [-3.20, 0.76, 2.12], [4.87, -1.40, 3.55]])
# b0 = np.array([[1.53], [-0.73], [2.39]])
# W1 = np.array([[2.10, -4.50, 3.00], [1.71, -2.38, 0.85], [3.41, 1.25, -0.94]])
# b1 = np.array([[-1.11], [2.46], [3.60]])
# W2 = np.array([[1.85, -3.25, 2.42], [0.93, 4.17, -2.83]])
# b2 = np.array([[1.21], [-1.79]])

# layer0 = Layer(W0, b0, activation='re
# layer1 = Layer(W1, b1, activation='relu')
# layer2 = Layer(W2, b2, activation='id')

# layers = [layer0, layer1, layer2]

# network = Network(layers) 

intermediate_network = copy.deepcopy(network)

quant_config = {
        "Fw": 4,  # Weight quantization factor
        "Fb": 4,  # Bias quantization factor
        "Fin": 4,  # Input quantization factor
        "Fh": 4,  # Input quantization factor
        "Clb_in": -100,  # Lower bound for input
        "Cub_in": 100,  # Upper bound for input
        "Clb_w": -100,  # Lower bound for weights
        "Cub_w": 100,   # Upper bound for weights
        "Clb_b": -100,  # Lower bound for biases
        "Cub_b": 100,   # Upper bound for biases
        "Cub_h": 100   # Upper bound for hidden layer activations 
    }

quantized_network = intermediate_network.quantize(quant_config, version = 2)

################################################################
# INPUT                                                        #
################################################################

# r = 0.5

# input_box = generate_input_box(dim, r)


input_box = np.array([
        [0.5, 1.5],  
        [1.5, 2.5],
        [-1.5, -0.5]])


print(input_box)

################################################################
# TESTS                                                        #
################################################################

n = 10000000

print_network_parameters(network)

output_box_15 = network.propagate_vertices_random(input_box, n)
print(f"Propagation of random vertices through NN:\n {output_box_15}")

output_box_2, zonotopeNN = network.propagate_zonotope(input_box)
print(f"Propagation of zonotope through NN:\n {output_box_2}")

output_box_2C_std = network.propagate_constrained_zonotope(input_box, 'standard')
print(f"Propagation of constrained zonotope (standard) through NN:\n {output_box_2C_std}")

output_box_2C_rect = network.propagate_constrained_zonotope(input_box, 'rectangle')
print(f"Propagation of constrained zonotope (rectangle) through NN:\n {output_box_2C_rect}")

print_network_parameters(quantized_network)

output_box_35 = quantized_network.propagate_vertices_random(input_box, n)
print(f"Propagation of random vertices through QNN:\n {output_box_35}")

output_box_4, zonotopeQNN = quantized_network.propagate_zonotope(input_box)
print(f"Propagation of zonotope through QNN:\n {output_box_4}")

output_box_4C_std = quantized_network.propagate_constrained_zonotope(input_box,'standard')
print(f"Propagation of constrained zonotope (standard) through QNN:\n {output_box_4C_std}")

output_box_4C_rect = quantized_network.propagate_constrained_zonotope(input_box,'rectangle')
print(f"Propagation of constrained zonotope (rectangle) through QNN:\n {output_box_4C_rect}")

abstraction_error_zonotope = abstraction_error(network, zonotopeNN, input_box)
print(f"Abstraction error zonotope of zonotope propagated through NN:\nWeight:{abstraction_error_zonotope.W}\nbias:{abstraction_error_zonotope.b}")
print(f"Its concretization:\n{abstraction_error_zonotope.concretize()}")

abstraction_error_zonotope_quantized = abstraction_error(quantized_network, zonotopeQNN, input_box)
print(f"Abstraction error zonotope of zonotope propgated through QNN:\nWeight:{abstraction_error_zonotope_quantized.W}\nbias:{abstraction_error_zonotope_quantized.b}")
print(f"Its concretization=:\n{abstraction_error_zonotope_quantized.concretize()}")

quantization_error_zonotope = quantization_error(network, quantized_network, zonotopeNN, zonotopeQNN, input_box, 0.5, 0.5)
print(f"Quantization error zonotope:\nWeight:{quantization_error_zonotope.W}\nbias:{quantization_error_zonotope.b}")
print(f"Concretized quantization error zonotope:\n{quantization_error_zonotope.concretize()}")




