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
from zonotope import Zonotope
from constrained_zonotope import ConstrainedZonotope
from oracle import sample_vectors_from_box
from abstract import abstract_to_vertices
import numpy as np
import matplotlib.pyplot as plt

###################################################################################################################
# TEST1: Analyzing the abstraction error between zonotopes and constrained zonotopes in quantized neural networks #
###################################################################################################################

def zonotopes_abstraction_performance_QNN(structures, samples, configuration, version):

    barplot = np.zeros((len(structures), 3))

    for i in range(len(structures)):

        zonotope_error = np.zeros(samples)
        constrained_zonotope_std_error = np.zeros(samples)
        constrained_zonotope_rect_error = np.zeros(samples)

        for j in range(samples):

            network = generate_network(structures[i])
            quantized_network = network.quantize(configuration, version = 2)

            input_box = generate_input_box(3, 0.5)

            output_box_rv = quantized_network.propagate_vertices_random(input_box, n = 10000)
            output_box_zonotope, _ = quantized_network.propagate_zonotope(input_box)
            output_box_constrained_zonotope_std = quantized_network.propagate_constrained_zonotope(input_box, 'standard')
            output_box_constrained_zonotope_rect = quantized_network.propagate_constrained_zonotope(input_box, 'rectangle')

            zonotope_error[j] = abs(output_box_rv[0][0] - output_box_zonotope[0][0]) + abs(output_box_rv[0][1] - output_box_zonotope[0][1])
            constrained_zonotope_std_error[j] = abs(output_box_rv[0][0] - output_box_constrained_zonotope_std[0][0]) + abs(output_box_rv[0][1] - output_box_constrained_zonotope_std[0][1])
            constrained_zonotope_rect_error[j] = abs(output_box_rv[0][0] - output_box_constrained_zonotope_rect[0][0]) + abs(output_box_rv[0][1] - output_box_constrained_zonotope_rect[0][1])


        barplot[i][0] = np.mean(zonotope_error)
        barplot[i][1] = np.mean(constrained_zonotope_std_error)
        barplot[i][2] = np.mean(constrained_zonotope_rect_error)

    x = np.arange(len(structures))

    groups = np.zeros(len(structures)) 
    for k in range(len(structures)): groups[k] = sum(structures[k]) - structures[k][0]

    bar_width = 0.3

    fig, ax = plt.subplots()

    categories = ['Zonotope', 'Constrained Zonotope (standard abstraction)',  'Constrained Zonotope (rectangle abstraction)']

    for i in range(3): ax.bar(x + i * bar_width, barplot[:, i], width=bar_width, label = categories[i])

    ax.set_xticks(x + bar_width / 3)
    ax.set_xticklabels([f'{int(groups[i])}' for i in range(len(structures))]) 
    ax.set_ylabel("Abstraction error")
    ax.set_xlabel("Number of neurons")
    ax.legend()

    plt.show()



structure1 = [3,2,2,2,1] # 7 neurons
structure2 = [3,3,2,2,1] # 8 neurons
structure3 = [3,3,3,2,1] # 9 neurons
structure4 = [3,3,3,3,1] # 10 neurons
structure5 = [3,4,3,3,1] # 11 neurons

structures = [structure1, structure2, structure3, structure4, structure5]

samples = 100

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

version = 2

# zonotopes_abstraction_performance_QNN(structures, samples, quant_config, version)

###############################################################################################################################################
# TEST2: Analyzing the qantization error as a function of the number of neurons in a neural network                                           #
###############################################################################################################################################


def quantization_error_analysis(structures, samples, configuration, version):

    interval_plot = np.zeros((len(structures), 2))

    for i in range(len(structures)):

        quantization_error_vector = np.zeros((samples, 2))

        for j in range(samples):

            network = generate_network(structures[i])
            intermediate_network = copy.deepcopy(network)
            quantized_network = intermediate_network.quantize(configuration, version = 2)

            input_box = generate_input_box(3, 0.5)

            output_box_zonotope, zonotopeNN = network.propagate_zonotope(input_box)
            output_box_zonotope_quantized, zonotopeQNN = quantized_network.propagate_zonotope(input_box)

            quantization_error_zonotope = quantization_error(network,quantized_network, zonotopeNN, zonotopeQNN, input_box, 0.5, 0.5)

            quantization_error_vector[j] = quantization_error_zonotope.concretize()

        interval_plot[i] = np.mean(quantization_error_vector, axis=0)

    m = interval_plot.shape[0] 
    y_positions = np.arange(m)  

    groups = np.zeros(len(structures)) 
    for k in range(len(structures)): groups[k] = sum(structures[k]) - structures[k][0]

    fig, ax = plt.subplots()

    for i in range(m): ax.plot([y_positions[i], y_positions[i]], interval_plot[i], marker="_", markersize=10, color='r')

    ax.set_xticks(y_positions)
    ax.set_xticklabels([f"{int(groups[i])}" for i in range(m)])
    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Quantization error")

    plt.show()


structure1 = [3,2,2,2,1] # 7 neurons
structure2 = [3,3,2,2,1] # 8 neurons
structure3 = [3,3,3,2,1] # 9 neurons
structure4 = [3,3,3,3,1] # 10 neurons
structure5 = [3,4,3,3,1] # 11 neurons

structures = [structure1, structure2, structure3, structure4, structure5]

samples = 10000

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

version = 2

quantization_error_analysis(structures, samples, quant_config, version)