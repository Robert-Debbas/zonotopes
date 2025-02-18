import numpy as np
from itertools import combinations
from itertools import product
from Zonotope import Zonotope
from CZonotope import ConstrainedZonotope

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

def abstract_to_zonotope(box):
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

def abstract_to_constrained_zonotope(box):
    """
    Converts a box into a zonotope.

    Parameters:
    - box: A NumPy array of shape (d, 2), where each row represents [lower_bound, upper_bound].

    Returns:
    - ConstraintedZonotope: A constrainted zonotope representing the abstracted box.
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
    return ConstrainedZonotope(W, center, [])


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
    zonotope = abstract_to_zonotope(input_box)

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

    final_zonotope = Zonotope(W_output.reshape(1, -1), np.array([b_output]))

    # Return the output zonotope
    return final_zonotope


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

    zonotope = abstract_to_zonotope(quantized_input)

    W_in = zonotope.W
    b_in = zonotope.b

    W_hidden = (2 ** (Fh - Fw - Fin)) * np.dot(quantized_weights_hidden, W_in)
    b_hidden = (2 ** (Fh - Fw - Fin)) * np.dot(quantized_weights_hidden, b_in) + (2 ** (Fh - Fb)) * quantized_bias_hidden

    zonotope = Zonotope(W_hidden, b_hidden)
    zonotope = zonotope.abstract_round() # TO TEST ON DIFFERENT VALUES OF LAMBDA
    zonotope = zonotope.abstract_clamp(Cub_h) 

    W_last = zonotope.W
    b_last = zonotope.b

    W_out = (2 ** (- Fw - Fh)) * np.dot(quantized_weights_output, W_last)
    b_out = (2 ** (- Fw - Fh)) * np.dot(quantized_weights_output, b_last) + (2 ** (- Fb - Fh)) * quantized_bias_output

    final_zonotope = Zonotope(W_out.reshape(1, -1), np.array([b_out]))

    return final_zonotope


def forward_pass_constrained_zonotope(input_box, weights_hidden, bias_hidden, weights_output, bias_output):
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
    czonotope = abstract_to_constrained_zonotope(input_box)

    # Step 1: Propagate input zonotope through the hidden layer
    # Representation of input: z_in(eps) = W_in * eps + b_in
    W_in, b_in = czonotope.W, czonotope.b

    # Hidden layer computation: z_hidden(eps) = ReLU(W_hidden * z_in + b_hidden)
    # z_hidden = W_hidden * (W_in * eps + b_in) + b_hidden
    W_hidden = np.dot(weights_hidden, W_in)  # Propagate weights
    b_hidden = np.dot(weights_hidden, b_in) + bias_hidden  # Propagate biases
    hidden_czonotope = ConstrainedZonotope(W_hidden, b_hidden, czonotope.constraints)

    hidden_czonotope = hidden_czonotope.abstract_ReLU_constrained()  # Apply ReLU abstraction
    print(hidden_czonotope.constraints)

    # Step 2: Propagate through the output layer
    # z_out(eps) = W_output * z_hidden + b_output
    W_hidden_out, b_hidden_out = hidden_czonotope.W, hidden_czonotope.b
    W_output = np.dot(weights_output, W_hidden_out)  # Propagate weights to output
    b_output = np.dot(weights_output, b_hidden_out) + bias_output  # Add biases

    final_czonotope = ConstrainedZonotope(W_output.reshape(1, -1), np.array([b_output]), hidden_czonotope.constraints)

    # Return the output zonotope
    return final_czonotope

def forward_pass_quantized_constrained_zonotope(input_box, weights_hidden, bias_hidden, weights_output, bias_output, quant_config):
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

    czonotope = abstract_to_constrained_zonotope(quantized_input)

    W_in = czonotope.W
    b_in = czonotope.b

    W_hidden = (2 ** (Fh - Fw - Fin)) * np.dot(quantized_weights_hidden, W_in)
    b_hidden = (2 ** (Fh - Fw - Fin)) * np.dot(quantized_weights_hidden, b_in) + (2 ** (Fh - Fb)) * quantized_bias_hidden

    czonotope = ConstrainedZonotope(W_hidden, b_hidden, czonotope.constraints)
    czonotope = czonotope.abstract_round() # TO TEST ON DIFFERENT VALUES OF LAMBDA
    czonotope = czonotope.abstract_clamp(Cub_h) 

    W_last = czonotope.W
    b_last = czonotope.b

    W_out = (2 ** (- Fw - Fh)) * np.dot(quantized_weights_output, W_last)
    b_out = (2 ** (- Fw - Fh)) * np.dot(quantized_weights_output, b_last) + (2 ** (- Fb - Fh)) * quantized_bias_output

    final_czonotope = ConstrainedZonotope(W_out.reshape(1, -1), np.array([b_out]), czonotope.constraints)

    return final_zonotope


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
        "Cub_h": 100   # Upper bound for hidden layer activations # USE 70 TO TEST ABSTRACT CLAMP
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

    print("Traditional Neural Network :")
    print("Output zonotope (W):", output_zonotope_traditional.W)
    print("Output zonotope (b):", output_zonotope_traditional.b)
    print("Output concretized zonotope in NN:", output_zonotope_traditional.concretize())

    output_zonotope_quantized = forward_pass_quantized_zonotope(input_box, weights_hidden, bias_hidden, weights_output, bias_output, quant_config)

    print("Quantized Neural Network :")
    print("Output Zonotope (W):", output_zonotope_quantized.W)
    print("Output Zonotope (b):", output_zonotope_quantized.b)
    print("Output concretized zonotope in QNN:", output_zonotope_quantized.concretize())

    output_czonotope_traditional = forward_pass_constrained_zonotope(input_box, weights_hidden, bias_hidden, weights_output, bias_output)

    print("Traditional Neural Network :")
    print("Output czonotope (W):", output_czonotope_traditional.W)
    print("Output czonotope (b):", output_czonotope_traditional.b)
    print("Output concretized czonotope in NN:", output_czonotope_traditional.concretize())

    output_czonotope_quantized = forward_pass_quantized_constrained_zonotope(input_box, weights_hidden, bias_hidden, weights_output, bias_output, quant_config)

    print("Quantized Neural Network :")
    print("Output czonotope (W):", output_czonotope_quantized.W)
    print("Output czonotope (b):", output_czonotope_quantized.b)
    print("Output concretized czonotope in QNN:", output_czonotope_quantized.concretize())





