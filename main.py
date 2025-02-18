import numpy as np
from itertools import combinations
from itertools import product
import copy
from network import Layer 
from network import Network
from zonotope import Zonotope
from constrained_zonotope import ConstrainedZonotope

################################################################
# CONSTRUCT NETWORKS                                           #             
################################################################


W0 = np.array([[1.25, -2.31, 4.52], [-3.20, 0.76, 2.12], [4.87, -1.40, 3.55]])
b0 = np.array([[1.53], [-0.73], [2.39]])
W1 = np.array([[2.10, -4.50, 3.00], [1.71, -2.38, 0.85], [3.41, 1.25, -0.94]])
b1 = np.array([[-1.11], [2.46], [3.60]])
W2 = np.array([[1.85, -3.25, 2.42], [0.93, 4.17, -2.83]])
b2 = np.array([[1.21], [-1.79]])

layer0 = Layer(W0, b0, activation='relu')
layer1 = Layer(W1, b1, activation='relu')
layer2 = Layer(W2, b2, activation='id')

layers = [layer0, layer1, layer2]

network = Network(layers) 

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

quantized_network = intermediate_network.quantize(quant_config)

################################################################
# INPUT                                                        #
################################################################

input_box = np.array([
        [0.5, 1.5],  
        [1.5, 2.5],
        [-1.5, -0.5]
    ])

################################################################
# TESTS                                                        #
################################################################

output_box_1 = network.propagate_vertices(input_box)
print(f"Propagation of vertices through NN : {output_box_1}")

output_box_2 = network.propagate_zonotope(input_box)
print(f"Propagation of zonotope through NN : {output_box_2}")

output_box_3 = quantized_network.propagate_vertices(input_box)
print(f"Propagation of vertices through QNN : {output_box_3}")

output_box_4 = quantized_network.propagate_zonotope(input_box)
print(f"Propagation of zonotope through QNN : {output_box_4}")


