import numpy as np
from itertools import combinations
from itertools import product
from zonotope import Zonotope
from constrained_zonotope import ConstrainedZonotope
from abstract import clamp
from abstract import abstract_to_vertices
from abstract import abstract_to_constrained_zonotope
from abstract import concretize_to_box
from abstract import abstract_to_zonotope
from abstract import relu_interval_propagation
from abstract import linear_interval_propagation
from abstract import clamp_interval_propagation
from oracle import sample_vectors_from_box


class Layer:

    def __init__(self, weights, biases, activation, lower_bound = None, upper_bound = None):

        self.weights = np.array(weights)
        self.biases = np.array(biases)

        if self.weights.shape[0] != self.biases.shape[0]:
            raise ValueError("Number of rows in W must be equal to the number of coefficients in b.")

        self.activation = activation

        if lower_bound: self.lower_bound = lower_bound
        if upper_bound: self.upper_bound = upper_bound

    def propagate(self, vector):

        vector = (self.weights @ vector.T + self.biases).T

        if self.activation == 'relu': vector = np.maximum(0, vector)
        elif self.activation == 'clamp': vector = clamp(np.floor(vector), self.lower_bound, self.upper_bound)

        return vector

    def propagate_box(self, box):

        box = linear_interval_propagation(box, self.weights, self.biases)

        if self.activation == 'relu': box = relu_interval_propagation(box)
        elif self.activation == 'clamp': box = clamp_interval_propagation(np.floor(box), self.lower_bound, self.upper_bound)

        return box

    def propagate_vertices(self, vertices):
        
        vertices = np.dot(self.weights, vertices.T).T + self.biases.T

        if self.activation == 'relu': vertices = np.maximum(0, vertices)
        elif self.activation == 'clamp': vertices = clamp(np.floor(vertices), self.lower_bound, self.upper_bound)

        return vertices

    def propagate_zonotope(self, zonotope):
        
        zonotope = zonotope.linear_transformation(self.weights, self.biases)     

        if self.activation == 'relu': zonotope = zonotope.abstract_ReLU()
        elif self.activation == 'clamp': zonotope = zonotope.abstract_floor().abstract_clamp(self.upper_bound)

        return zonotope

    def propagate_constrained_zonotope(self, czonotope, version):
        
        czonotope = czonotope.linear_transformation(self.weights, self.biases)

        if self.activation == 'relu': czonotope = czonotope.abstract_ReLU(version)
        elif self.activation == 'clamp': czonotope = czonotope.abstract_floor().abstract_clamp(self.upper_bound, version)

        return czonotope
        


class Network:

    def __init__(self, layers):

        if not all(isinstance(layer, Layer) for layer in layers):
            raise TypeError("All elements in layers must be instances of the Layer class.")

        for i in range(len(layers) - 1):
            if layers[i].weights.shape[0] != layers[i + 1].weights.shape[1]:
                raise ValueError("Number of columns in W of one layer must match the number of rows in W of the next layer.")

        self.layers = layers

        self.quantized = False

    def quantize(self, configuration, version):

        self.quantized = True 

        Fw, Fb, Fin, Fh = configuration["Fw"], configuration["Fb"], configuration["Fin"], configuration["Fh"]
        Clb_in, Cub_in = configuration["Clb_in"], configuration["Cub_in"]
        Clb_w, Cub_w = configuration["Clb_w"], configuration["Cub_w"]
        Clb_b, Cub_b = configuration["Clb_b"], configuration["Cub_b"]
        Cub_h = configuration["Cub_h"]

        if version == 1: 
            exponent1 = Fh - Fw - Fin
            exponent2 = Fh - Fb
        else: 
            exponent1 = - Fw - Fin
            exponent2 = - Fb


        self.layers[0].weights = clamp(np.floor(self.layers[0].weights * (2 ** Fw)), Clb_w, Cub_w)
        self.layers[0].weights = (2 ** (exponent1)) * self.layers[0].weights

        self.layers[0].biases = clamp(np.floor(self.layers[0].biases * (2 ** Fb)), Clb_b, Cub_b)
        self.layers[0].biases = (2 ** (exponent2)) * self.layers[0].biases

        if self.layers[0].activation == 'relu': 
                self.layers[0].activation = 'clamp'
                self.layers[0].lower_bound = 0
                self.layers[0].upper_bound = Cub_h

        if version == 1: 
            exponent1 = Fh - Fw 
            exponent2 = Fh - Fb
        else: 
            exponent1 = - Fw 
            exponent2 = - Fb

        for i in range(1, len(self.layers) - 1):

            self.layers[i].weights = clamp(np.floor(self.layers[i].weights * (2**Fw)), Clb_w, Cub_w)
            self.layers[i].weights = (2**(exponent1)) * self.layers[i].weights
            self.layers[i].biases = clamp(np.floor(self.layers[i].biases * (2**Fb)), Clb_b, Cub_b)
            self.layers[i].biases = (2**(exponent2)) * self.layers[i].biases

            if self.layers[i].activation == 'relu': 
                self.layers[i].activation = 'clamp'
                self.layers[i].lower_bound = 0
                self.layers[i].upper_bound = Cub_h

        if version == 1: 
            exponent1 = - Fw - ( len(self.layers) - 2) * Fh
            exponent2 = - Fb - ( len(self.layers) - 2) * Fh
        else: 
            exponent1 = - Fw 
            exponent2 = - Fb

        self.layers[-1].weights = clamp(np.floor(self.layers[-1].weights * (2**Fw)), Clb_w, Cub_w)
        self.layers[-1].weights = (2**(exponent1)) * self.layers[-1].weights

        self.layers[-1].biases = clamp(np.floor(self.layers[-1].biases * (2**Fb)), Clb_b, Cub_b)
        self.layers[-1].biases = (2**(exponent2)) * self.layers[-1].biases

        if self.layers[-1].activation == 'relu': 
                self.layers[-1].activation = 'clamp'
                self.layers[-1].lower_bound = 0
                self.layers[-1].upper_bound = Cub_h
            
        I = np.eye(self.layers[0].weights.shape[1])
        b0 = np.zeros((self.layers[0].weights.shape[1] , 1))

        input_layer = Layer((2**(Fin)) * I, b0, 'clamp', Clb_in, Cub_in)

        self.layers = [input_layer] + self.layers
        
        return self

    def propagate(self, vector):
        
        for i in range(len(self.layers)): vector = self.layers[i].propagate(vector)

        return vector.T

    def propagate_box(self, box):

        for i in range(len(self.layers)): box = self.layers[i].propagate_box(box)

        return box

    def propagate_vertices(self, box):

        vertices = abstract_to_vertices(box)
        for i in range(len(self.layers)): vertices = self.layers[i].propagate_vertices(vertices)
        out_box = concretize_to_box(vertices)

        return out_box
     
    def propagate_vertices_random(self, box, n):

        vertices = np.vstack((abstract_to_vertices(box), sample_vectors_from_box(box,n)))
        for i in range(len(self.layers)): vertices = self.layers[i].propagate_vertices(vertices)
        out_box = concretize_to_box(vertices)

        return out_box

    def propagate_zonotope(self, box):

        start = 0

        if self.quantized == True:

            vertices = abstract_to_vertices(box)
            vertices = self.layers[0].propagate_vertices(vertices)
            box = concretize_to_box(vertices)

            start = 1 

        zonotope = abstract_to_zonotope(box)

        for i in range(start, len(self.layers)): 
            zonotope = self.layers[i].propagate_zonotope(zonotope)

        out_box = zonotope.concretize()

        return out_box, zonotope
    
    def propagate_constrained_zonotope(self, box, version):
        
        start = 0

        if self.quantized == True:

            vertices = abstract_to_vertices(box)
            vertices = self.layers[0].propagate_vertices(vertices)
            box = concretize_to_box(vertices)

            start = 1 

        czonotope = abstract_to_constrained_zonotope(box)

        for i in range(start, len(self.layers)): czonotope = self.layers[i].propagate_constrained_zonotope(czonotope, version)

        out_box = czonotope.final_concretize()

        return out_box

def generate_network(structure):
    """
    Generates a deep forward neural network with ReLU activations in all layers
    except for the output layer, which has an identity activation.

    :param structure: List representing the number of neurons in each layer
                      e.g., [2, 2, 1] -> Input layer with 2 neurons, one hidden layer with 2 neurons, output layer with 1 neuron
    :return: A Network object
    """

    layers = []

    for i in range(len(structure) - 1):

        W = np.random.uniform(-5, 5, (structure[i + 1], structure[i])) 
        b = np.random.uniform(-5, 5, (structure[i + 1], 1))
        activation = 'relu' if i < len(structure) - 2 else 'id' 
        layers.append(Layer(W, b, activation))

    return Network(layers)

def generate_input_box(dim, r):
    """
    Generates a random dim-dimensional input box centered around a randomly generated point.
    The box is constructed with a given radius r.

    :param dim: Dimension of the input box
    :param r: Radius of the box
    :return: A NumPy array of shape (dim, 2) representing the input box
    """
    center = np.random.uniform(-5, 5, dim)  # Random center point within [-5, 5]
    
    # Compute lower and upper bounds based on radius r
    lower_bounds = center - r  
    upper_bounds = center + r

    # Format as a NumPy array with shape (dim, 2)
    input_box = np.column_stack((lower_bounds, upper_bounds))
    
    return input_box

def print_network_parameters(network):
    """
    Prints the weights and biases of a given Network object.

    :param network: A Network object containing layers with weights and biases
    """
    for idx, layer in enumerate(network.layers):
        print(f"Layer {idx}:")
        print("Weights:\n", layer.weights)
        print("Biases:\n", layer.biases)
        print("Activation:", layer.activation)
        print("-" * 40)

def structure_from_network(NN):

    structure = []

    for i in range(len(NN.layers)): structure.append(len(NN.layers[i].weights[0]))

    structure.append(len(NN.layers[-1].weights))

    return structure

def center_from_box(box):

    center = np.zeros((1, len(box)))

    for i in range(len(box)): center[0][i] = (box[i][0] + box[i][1])/2

    return center


def quantization_error(NN, QNN, zonotopeNN, zonotopeQNN, input_box, gamma1, gamma2): 

    structure = structure_from_network(NN)
    center = center_from_box(input_box)

    W, b = zonotopeNN.W, zonotopeNN.b
    Wq, bq = zonotopeQNN.W, zonotopeQNN.b
    We, be = np.zeros((len(Wq), len(Wq[0]))), np.zeros_like(zonotopeQNN.b)

    WT = W.T
    WqT = Wq.T
    WeT = We.T

    start = 0
    start_qe = 0

    for i in range(structure[0]): WeT[start_qe + i] = WqT[start_qe + i] - WT[start + i] 
    
    start += structure[0]
    start_qe += structure[0]

    for i in range(1, len(structure) - 1): 
        for j in range(structure[i]): WeT[start_qe + j] = WqT[start_qe + j]
        start_qe += structure[i]
        for j in range(structure[i]): WeT[start_qe + j] = WqT[start_qe + j] - WT[start + j] 
        start += structure[i]
        start_qe += structure[i]
            
    b_prime = NN.propagate(center)
    bq_prime = QNN.propagate(center)

    We = WeT.T
    be = (gamma1 * bq_prime + (1 - gamma1) * bq) - (gamma2 * b_prime + (1 - gamma2) * b)

    return Zonotope(We, be)


def abstraction_error(network, zonotope, input_box):

    structure = structure_from_network(network)

    W, b = zonotope.W, zonotope.b

    center = center_from_box(input_box)

    dim_input = structure[0]
    
    print(center)

    b_prime = network.propagate(center)

    return Zonotope(W[:,dim_input:], b_prime - b)


def quantization_error_vertices(NN, QNN, input_box, samples):

    structure = structure_from_network(NN)

    vertices = sample_vectors_from_box(input_box, samples)

    errors = np.zeros((structure[-1], len(vertices)))

    for i in range(len(vertices)): 

        output_NN = NN.propagate(np.array([vertices[i]]))
        output_QNN = QNN.propagate(np.array([vertices[i]]))
        
        errors[:,i] = output_QNN - output_NN

    upper_error = np.max(errors, axis = 1)
    lower_error = np.min(errors, axis = 1)

    error_bounds = np.hstack((lower_error, upper_error))

    return error_bounds


    





    

