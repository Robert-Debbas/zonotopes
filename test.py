import numpy as np
from itertools import combinations
from itertools import product

def forward_pass_box(input_box, weights_hidden, bias_hidden, weights_output, bias_output):
    """
    Perform a forward pass through a simple neural network with a box input
    by propagating its vertices.

    Parameters:
        input_box (array): Input box with shape (m, 2), where each row is [min, max].
        weights_hidden (array): Weights for the hidden layer, shape (n_hidden, m).
        bias_hidden (array): Biases for the hidden layer, shape (n_hidden,).
        weights_output (array): Weights for the output layer, shape (n_hidden,).
        bias_output (float): Bias for the output neuron.

    Returns:
        list: The output interval [min, max] of the neural network.
    """

    # Generate all vertices of the input box
    vertices = np.array(list(product(*input_box)))  

    # Propagate the vertices through the network
    vertices = (np.dot(weights_hidden, vertices.T)).T + bias_hidden
    vertices = np.maximum(0, vertices)  # ReLU activation
    vertices = (np.dot(weights_output, vertices.T)).T + bias_output  

    # Determine the output interval
    output_min = np.min(vertices)
    output_max = np.max(vertices)

    return [output_min, output_max]


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

def forward_pass_quantized_box(input_box, weights_hidden, bias_hidden, weights_output, bias_output, quant_config):
    """
    Perform a forward pass through a quantized neural network with interval inputs.

    Parameters:
        input_box (array): Input box with shape (2, 2), where each entry is [a, b].
        weights_hidden (array): Weights for the hidden layer, shape (2, 2).
        bias_hidden (array): Biases for the hidden layer, shape (2,).
        weights_output (array): Weights for the output layer, shape (2,).
        bias_output (float): Bias for the output neuron.
        quant_config (dict): Quantization configuration containing:
            - "Fw": Weight quantization factor.
            - "Fb": Bias quantization factor.
            - "Fin": Input quantization factor.
            - "Cub_w": Upper bound for quantized weights.
            - "Clb_w": Lower bound for quantized weights.
            - "Cub_b": Upper bound for quantized biases.
            - "Clb_b": Lower bound for quantized biases.
            - "Cub_h": Upper bound for hidden layer activations.

    Returns:
        tuple: The output interval [min, max] of the quantized neural network.
    """
    # Unpack quantization configuration
    Fw, Fb, Fin, Fh = quant_config["Fw"], quant_config["Fb"], quant_config["Fin"], quant_config["Fh"]
    Clb_in, Cub_in = quant_config["Clb_in"], quant_config["Cub_in"]
    Clb_w, Cub_w = quant_config["Clb_w"], quant_config["Cub_w"]
    Clb_b, Cub_b = quant_config["Clb_b"], quant_config["Cub_b"]
    Cub_h = quant_config["Cub_h"]

    # Quantize weights and biases
    quantized_input = clamp(np.round(input_box * (2**Fin)), Clb_in, Cub_in)
    quantized_weights_hidden = clamp(np.round(weights_hidden * (2**Fw)), Clb_w, Cub_w)
    quantized_bias_hidden = clamp(np.round(bias_hidden * (2**Fb)), Clb_b, Cub_b)
    quantized_weights_output = clamp(np.round(weights_output * (2**Fw)), Clb_w, Cub_w)
    quantized_bias_output = clamp(np.round(bias_output * (2**Fb)), Clb_b, Cub_b)

    # Propagate the vertices through the quantized network
    vertices = np.array(list(product(*quantized_input)))
    vertices = np.round((2**(Fh - Fw - Fin)) * ((np.dot(quantized_weights_hidden, vertices.T)).T) + (2**(Fh - Fb)) * quantized_bias_hidden)
    vertices = clamp(vertices, 0, Cub_h)
    vertices = (2 ** (- Fw - Fh)) * ((np.dot(quantized_weights_output, vertices.T)).T) + (2**(- Fb - Fh)) * quantized_bias_output

    # Determine the output interval
    output_min = np.min(vertices)
    output_max = np.max(vertices)

    return [output_min, output_max]



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
    
    def evaluate(self, eps):
        """
        Evaluate z(eps) = W eps + b for a given eps.

        Parameters:
        - eps: numpy array of shape (m,), the noise vector.

        Returns:
        - numpy array of shape (d,), the zonotope value.
        """
        eps = np.array(eps)  # Ensure eps is a numpy array
        if eps.shape[0] != self.W.shape[1]:
            raise ValueError("The dimension of eps must match the number of columns in W.")
        return np.dot(self.W, eps) + self.b

    def bounds(self):
        """
        Compute the bounds of the zonotope and return as an array of intervals.

        Returns:
        - A NumPy array of shape (d, 2), where each row is [lower_bound, upper_bound].
        """
        abs_W = np.abs(self.W)
        lower_bounds = self.b - np.sum(abs_W, axis=1)
        upper_bounds = self.b + np.sum(abs_W, axis=1)
        return lower_bounds, upper_bounds
    
    def concretize(self):
        """
        Compute the bounds of the zonotope and return as an array of intervals.

        Returns:
        - A NumPy array of shape (d, 2), where each row is [lower_bound, upper_bound].
        """
        abs_W = np.abs(self.W)
        lower_bounds = self.b - np.sum(abs_W, axis=1)
        upper_bounds = self.b + np.sum(abs_W, axis=1)
        intervals = np.stack((lower_bounds, upper_bounds), axis=1)
        return intervals

    def abstract_ReLU(self):
        """
        Abstracts the ReLU function over the current zonotope.

        Returns:
        - Zonotope: A new Zonotope object representing the abstracted ReLU.
        """
        l, u = self.bounds() 

        W_new = []
        b_new = []

        for i in range(self.W.shape[0]):
            if u[i] <= 0:
                # Fully negative case: y = 0
                W_new.append(np.hstack((np.zeros(self.W.shape[1]), 0)))  # Append 0 for the new noise dimension
                b_new.append(0)
            elif l[i] >= 0:
                # Fully positive case: y = x
                W_new.append(np.hstack((self.W[i], 0)))  # Append 0 for the new noise dimension
                b_new.append(self.b[i])
            else:
                # Unstable case: l[i] < 0 < u[i]
                a_param = -u[i] / (u[i] - l[i])
                b_param = -u[i] * l[i] / (u[i] - l[i])

                # Adjust W and add new noise symbol
                new_row = np.hstack((a_param * self.W[i], b_param / 2))
                W_new.append(new_row)
                b_new.append(a_param * self.b[i] + b_param / 2)

        W_new = np.array(W_new)
        b_new = np.array(b_new)

        return Zonotope(W_new, b_new)

    def abstract_clamp(self, C_ub):
        """
        Abstracts the clamp(x, 0, C^{ub}) function over the current zonotope.

        Parameters:
        - C_ub: float, the upper bound of the clamp function.

        Returns:
        - Zonotope: A new Zonotope object representing the abstracted clamp.
        """
        l, u = self.bounds()  # Get the bounds of the zonotope

        W_new = []
        b_new = []

        for i in range(self.W.shape[0]):
            if u[i] <= 0:
                # Fully negative case: clamp(x, 0, C^{ub}) = 0
                W_new.append(np.hstack((np.zeros(self.W.shape[1]), 0)))  # Append 0 for the new noise dimension
                b_new.append(0)
            elif l[i] >= C_ub:
                # Fully above C^{ub}: clamp(x, 0, C^{ub}) = C^{ub}
                W_new.append(np.hstack((np.zeros(self.W.shape[1]), 0)))  # No noise
                b_new.append(C_ub)
            elif l[i] >= 0 and u[i] <= C_ub:
                # Fully within the range [0, C^{ub}]: clamp(x, 0, C^{ub}) = x
                W_new.append(np.hstack((self.W[i], 0)))  # Append 0 for the new noise dimension
                b_new.append(self.b[i])
            else:
                # Mixed case: clamp(x, 0, C^{ub}) needs abstraction
                if l[i] < 0 and C_ub <= u[i]:
                    # Case: l <= 0 <= C^{ub} <= u
                    a_param = -u[i] / (u[i] - l[i])
                    b_param = -u[i] * l[i] / (u[i] - l[i])

                    # Adjust W and add new noise symbol
                    new_row = np.hstack((a_param * self.W[i], b_param / 2))
                    W_new.append(new_row)
                    b_new.append(a_param * self.b[i] + b_param / 2)
                elif l[i] < 0 and u[i] <= C_ub:
                    # Case: l <= 0 < u <= C^{ub}
                    a_param = 1
                    b_param = 0  # Because it clamps below 0
                    new_row = np.hstack((a_param * self.W[i], b_param / 2))
                    W_new.append(new_row)
                    b_new.append(a_param * self.b[i] + b_param / 2)
                elif l[i] >= 0 and C_ub < u[i]:
                    # Case: 0 <= l <= C^{ub} < u
                    a_param = C_ub / (u[i] - l[i])
                    b_param = l[i] * C_ub / (u[i] - l[i])

                    new_row = np.hstack((a_param * self.W[i], (C_ub - b_param) / 2))
                    W_new.append(new_row)
                    b_new.append(a_param * self.b[i] + (C_ub - b_param) / 2)

        W_new = np.array(W_new)
        b_new = np.array(b_new)

        return Zonotope(W_new, b_new)

    def abstract_round(self, lambda_val):
        """
        Abstracts the rounding function over the current zonotope.

        Parameters:
        - lambda_val: Regularization parameter for the least squares computation.

        Returns:
        - Zonotope: A new Zonotope object representing the abstracted round function.
        """
        W_new = np.zeros_like(self.W)
        b_new = np.zeros_like(self.b)

        # Iterate over each row of the zonotope
        for i in range(len(self.b)):
            # Extract the relevant weights and bias for row i
            wi = self.W[i, :]
            bi = self.b[i]

            # Define the matrix A
            A = np.array([
                [-1, -1, 1],
                [1, 1, 1]
            ])

            # Define the vector yi
            yi = np.array([
                [np.floor(-np.abs(wi[0]) - np.abs(wi[1]) + bi)], # QUESTIONABLE CHOICE
                [np.floor(np.abs(wi[0]) + np.abs(wi[1]) + bi) + 1]
            ])

            # Regularization: Identity matrix with the bottom-right coefficient set to 0
            I = np.eye(3)
            I[-1, -1] = 0

            # Compute vi
            AT = A.T
            vi = np.linalg.inv(AT @ A + lambda_val * I) @ AT @ yi

            # Update the coefficients for the new zonotope
            W_new[i, 0] = vi[0, 0]
            W_new[i, 1] = vi[1, 0]
            b_new[i] = vi[2, 0]
            
        # Return the new zonotope
        return Zonotope(W_new, b_new)

def abstract(box):
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


def forward_pass_zonotope(input_box, weights_hidden, bias_hidden, weights_output, bias_output):
    """
    Perform a forward pass through a simple neural network with a zonotope.

    Parameters:
        input_box : input_box (array): Input box with shape (m, 2), where each row is [min, max].
        weights_hidden (array): Weights for the hidden layer, shape (n_hidden, m).
        bias_hidden (array): Biases for the hidden layer, shape (n_hidden,).
        weights_output (array): Weights for the output layer, shape (n_hidden,).
        bias_output (float): Bias for the output neuron.

    Returns:
        Zonotope: The output zonotope of the neural network.
    """
    # Step 0: Abstract the input box into a zonotope
    zonotope = abstract(input_box)

    # Step 1: Propagate input zonotope through the hidden layer
    # Representation of input: z_in(eps) = W_in * eps + b_in
    W_in, b_in = zonotope.W, zonotope.b

    # Hidden layer computation: z_hidden(eps) = ReLU(W_hidden * z_in + b_hidden)
    # z_hidden = W_hidden * (W_in * eps + b_in) + b_hidden
    W_hidden = np.dot(weights_hidden, W_in)  # Propagate weights
    b_hidden = np.dot(weights_hidden, b_in) + bias_hidden  # Propagate biases

    hidden_zonotope = Zonotope(W_hidden, b_hidden)
    hidden_zonotope = hidden_zonotope.abstract_ReLU()  # Apply ReLU abstraction

    # Step 2: Propagate through the output layer
    # z_out(eps) = W_output * z_hidden + b_output
    W_hidden_out, b_hidden_out = hidden_zonotope.W, hidden_zonotope.b
    W_output = np.dot(weights_output, W_hidden_out)  # Propagate weights to output
    b_output = np.dot(weights_output, b_hidden_out) + bias_output  # Add biases

    # Return the output zonotope
    return Zonotope(W_output.reshape(1, -1), np.array([b_output]))


def forward_pass_quantized_zonotope(input_box, weights_hidden, bias_hidden, weights_output, bias_output, quant_config):
    """
    Perform a forward pass through a quantized neural network with zonotopes.

    Parameters:
        input_box : input_box (array): Input box with shape (m, 2), where each row is [min, max].
        weights_hidden (array): Weights for the hidden layer, shape (n_hidden, m).
        bias_hidden (array): Biases for the hidden layer, shape (n_hidden,).
        weights_output (array): Weights for the output layer, shape (n_hidden,).
        bias_output (float): Bias for the output neuron.
        quant_config (dict): Quantization configuration containing:
            - "Fw": Weight quantization factor.
            - "Fb": Bias quantization factor.
            - "Fin": Input quantization factor.
            - "Fh": Hidden activation quantization factor.
            - "Cub_w": Upper bound for quantized weights.
            - "Clb_w": Lower bound for quantized weights.
            - "Cub_b": Upper bound for quantized biases.
            - "Clb_b": Lower bound for quantized biases.
            - "Cub_h": Upper bound for hidden layer activations.

    Returns:
        Zonotope: The output zonotope of the quantized neural network.
    """

    # Unpack quantization configuration
    Fw, Fb, Fin, Fh = quant_config["Fw"], quant_config["Fb"], quant_config["Fin"], quant_config["Fh"]
    Clb_in, Cub_in = quant_config["Clb_in"], quant_config["Cub_in"]
    Clb_w, Cub_w = quant_config["Clb_w"], quant_config["Cub_w"]
    Clb_b, Cub_b = quant_config["Clb_b"], quant_config["Cub_b"]
    Cub_h = quant_config["Cub_h"]

    # Quantize weights and biases
    quantized_input = clamp(np.round(input_box * (2**Fin)), Clb_in, Cub_in)
    quantized_weights_hidden = clamp(np.round(weights_hidden * (2**Fw)), Clb_w, Cub_w)
    quantized_bias_hidden = clamp(np.round(bias_hidden * (2**Fb)), Clb_b, Cub_b)
    quantized_weights_output = clamp(np.round(weights_output * (2**Fw)), Clb_w, Cub_w)
    quantized_bias_output = clamp(np.round(bias_output * (2**Fb)), Clb_b, Cub_b)

    zonotope = abstract(quantized_input)

    W_in = zonotope.W
    b_in = zonotope.b

    W_hidden = (2 ** (Fh - Fw - Fin)) * np.dot(quantized_weights_hidden, W_in)
    b_hidden = (2 ** (Fh - Fw - Fin)) * np.dot(quantized_weights_hidden, b_in) + (2 ** (Fh - Fb)) * quantized_bias_hidden

    zonotope = Zonotope(W_hidden, b_hidden)
    zonotope = zonotope.abstract_round(0.1) # TO TEST ON DIFFERENT VALUES OF LAMBDA
    zonotope = zonotope.abstract_clamp(Cub_h) 

    W_last = zonotope.W
    b_last = zonotope.b

    W_out = (2 ** (- Fw - Fh)) * np.dot(quantized_weights_output, W_last)
    b_out = (2 ** (- Fw - Fh)) * np.dot(quantized_weights_output, b_last) + (2 ** (- Fb - Fh)) * quantized_bias_output

    return Zonotope(W_out.reshape(1, -1), np.array([b_out]))


if __name__ == "__main__":
    # Define input intervals and parameters
    input_box = np.array([
        [0.5, 1.5],  # Interval for the first input
        [1.5, 2.5]   # Interval for the second input
    ])

    # Weights for the hidden layer: shape (2, 2)
    weights_hidden = np.array([
        [3.98, 5.36],
        [4.02, 2.24]
    ])

    # Biases for the hidden layer: shape (2,)
    bias_hidden = np.array([6.72, -7.06])

    # Weights for the output layer: shape (2,)
    weights_output = np.array([0.26, 1.04])

    # Bias for the output layer: scalar
    bias_output = -2.92

    # Quantization configuration
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
        "Cub_h": 100    # Upper bound for hidden layer activations
    }

    # Perform the forward pass with the box
    output_interval = forward_pass_box(input_box, weights_hidden, bias_hidden, weights_output, bias_output)
    print(f"Output interval of the neural network: {output_interval}")

    # Perform the forward pass with quantized intervals
    output_interval = forward_pass_quantized_box(
        input_box, weights_hidden, bias_hidden, weights_output, bias_output, quant_config
    )
    print(f"Output interval of the quantized neural network: {output_interval}")

    output_zonotope_traditional = forward_pass_zonotope(input_box, weights_hidden, bias_hidden, weights_output, bias_output)

    # print("Traditional Neural Network :")
    # print("Output zonotope (W):", output_zonotope_traditional.W)
    # print("Output zonotope (b):", output_zonotope_traditional.b)
    print("Output concretized zonotope in NN:", output_zonotope_traditional.concretize())

    output_zonotope_quantized = forward_pass_quantized_zonotope(input_box, weights_hidden, bias_hidden, weights_output, bias_output, quant_config)

    # print("Quantized Neural Network :")
    # print("Output Zonotope (W):", output_zonotope_quantized.W)
    # print("Output Zonotope (b):", output_zonotope_quantized.b)
    print("Output concretized zonotope in QNN:", output_zonotope_quantized.concretize())


