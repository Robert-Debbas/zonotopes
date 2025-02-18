import numpy as np
from itertools import combinations
from itertools import product
from Zonotope import Zonotope
from CZonotope import ConstrainedZonotope
from abstract import clamp
from abstract import abstract_to_vertices
from abstract import concretize_to_box
from abstract import abstract_to_zonotope

class Layer:

    def __init__(self, weights, biases, activation, lower_bound = None, upper_bound = None):

        self.weights = np.array(weights)
        self.biases = np.array(biases)

        if self.weights.shape[0] != self.biases.shape[0]:
            raise ValueError("Number of rows in W must be equal to the number of coefficients in b.")

        self.activation = activation

        if lower_bound: self.lower_bound = lower_bound
        if upper_bound: self.upper_bound = upper_bound

    def propagate_vertices(self, vertices):
        
        vertices = np.dot(self.weights, vertices.T).T + self.biases.T

        if self.activation == 'relu': vertices = np.maximum(0, vertices)
        elif self.activation == 'clamp': vertices = np.minimum(np.maximum(np.round(vertices), self.lower_bound), self.upper_bound)

        return vertices
        
    def propagate_zonotope(self, zonotope):
        
        zonotope = zonotope.linear_transformation(self.weights, self.biases)

        if self.activation == 'relu': zonotope = zonotope.abstract_ReLU()
        elif self.activation == 'clamp': zonotope = zonotope.abstract_round().abstract_clamp(self.upper_bound)

        return zonotope

    def propagate_constrained_zonotope(self, czonotope):
        
        czonotope = czonotope.linear_transformation(self.weights, self.biases)

        if self.activation == 'relu': czonotope = czonotope.abstract_ReLU()
        elif self.activation == 'clamp': czonotope = czonotope.abstract_round().abstract_clamp(self.upper_bound)

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

    def quantize(self, configuration):

        self.quantized = True 

        Fw, Fb, Fin, Fh = configuration["Fw"], configuration["Fb"], configuration["Fin"], configuration["Fh"]
        Clb_in, Cub_in = configuration["Clb_in"], configuration["Cub_in"]
        Clb_w, Cub_w = configuration["Clb_w"], configuration["Cub_w"]
        Clb_b, Cub_b = configuration["Clb_b"], configuration["Cub_b"]
        Cub_h = configuration["Cub_h"]

        self.layers[0].weights = clamp(np.round(self.layers[0].weights * (2**Fw)), Clb_w, Cub_w)
        self.layers[0].weights = (2**(Fh - Fw - Fin)) * self.layers[0].weights

        self.layers[0].biases = clamp(np.round(self.layers[0].biases * (2**Fb)), Clb_b, Cub_b)
        self.layers[0].biases = (2**(Fh - Fb)) * self.layers[0].biases

        if self.layers[0].activation == 'relu': 
                self.layers[0].activation = 'clamp'
                self.layers[0].lower_bound = 0
                self.layers[0].upper_bound = Cub_h


        for i in range(1, len(self.layers)-1):

            self.layers[i].weights = clamp(np.round(self.layers[i].weights * (2**Fw)), Clb_w, Cub_w)
            self.layers[i].weights = (2**(Fh - Fw)) * self.layers[i].weights
            self.layers[i].biases = clamp(np.round(self.layers[i].biases * (2**Fb)), Clb_b, Cub_b)
            self.layers[i].biases = (2**(Fh - Fb)) * self.layers[i].biases

            if self.layers[i].activation == 'relu': 
                self.layers[i].activation = 'clamp'
                self.layers[i].lower_bound = 0
                self.layers[i].upper_bound = Cub_h

        self.layers[-1].weights = clamp(np.round(self.layers[-1].weights * (2**Fw)), Clb_w, Cub_w)
        self.layers[-1].weights = (2**(- Fw - ( len(self.layers) - 1) * Fh)) * self.layers[-1].weights

        self.layers[-1].biases = clamp(np.round(self.layers[-1].biases * (2**Fb)), Clb_b, Cub_b)
        self.layers[-1].biases = (2**(- Fb -( len(self.layers) - 1) * Fh)) * self.layers[-1].biases

        if self.layers[-1].activation == 'relu': 
                self.layers[-1].activation = 'clamp'
                self.layers[-1].lower_bound = 0
                self.layers[-1].upper_bound = Cub_h
            
        I = np.eye(self.layers[0].weights.shape[1])
        b0 = np.zeros((self.layers[0].weights.shape[1] , 1))

        input_layer = Layer((2**(Fin)) * I, b0, 'clamp', Clb_in, Cub_in)

        self.layers = [input_layer] + self.layers
        
        return self
     
    def propagate_vertices(self, box):

        vertices = abstract_to_vertices(box)
        for i in range(len(self.layers)):
            vertices = self.layers[i].propagate_vertices(vertices)
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

        for i in range(start, len(self.layers)): zonotope = self.layers[i].propagate_zonotope(zonotope)

        out_box = zonotope.concretize()

        return out_box
    
    def propagate_constrained_zonotope(self, czonotope):
        
        start = 0

        if self.quantized == True:

            vertices = abstract_to_vertices(box)
            vertices = self.layer[0].propagate_vertices(vertices)
            box = concretize_to_box(vertices)

            start = 1 

        czonotope = abstract_to_constrained_zonotope(box)

        for i in range(start, len(self.layers)): czonotope = self.layer[i].propagate_constrained_zonotope(czonotope)

        out_box = czonotope.concretize()

        return out_box

    


