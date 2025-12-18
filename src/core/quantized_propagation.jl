using ModelVerification
import ModelVerification: ReLU, Id, Network, Layer
using LazySets
using LinearAlgebra

"""
    quantization_error_zonotope(net::Network, quant_config, input_zonotope::Zonotope) -> Zonotope

Compute a zonotope over-approximation of the quantization error for a neural network.

This is the main algorithm that propagates quantization error through the network layers
using abstract interpretation with zonotopes. It tracks both the exact (real-valued)
computation and the quantized computation, maintaining their difference as a zonotope.

# Arguments
- `net::Network`: The neural network to verify (from ModelVerification.jl)
- `quant_config::Dict`: Quantization configuration with keys:
  - `:input => (strategy, bits, scaling)`: Input quantization parameters
  - `:weights => (strategy, bits, scaling)`: Weight quantization parameters
  - `:biases => (strategy, bits, scaling)`: Bias quantization parameters
  - `:activations => (strategy, bits, scaling)`: Activation quantization parameters
  where `strategy` is "p" (positive) or "pm" (plus-minus), `bits` is bit-width,
  and `scaling` is the fractional bit count
- `input_zonotope::Zonotope`: Input region as a zonotope

# Returns
- `Zonotope`: A zonotope over-approximating the quantization error at the output

# Algorithm
The algorithm maintains three zonotopes per layer:
- Z: Exact (real-valued) intermediate values
- Z_hat: Quantized intermediate values
- Z_tilda: The difference (quantization error)

For each layer, it:
1. Propagates through affine transformation with quantized and real weights/biases
2. Applies abstract ReLU or round-clamp operations
3. Accumulates the error propagation

# Example
```julia
net = load_acasxu_network_from_json("network.json")
quant_config = Dict(
    :input => ("pm", 8, 8),
    :weights => ("pm", 8, 6),
    :biases => ("pm", 8, 6),
    :activations => ("p", 8, 6)
)
input_center = [0.0, 0.0, 0.0, 0.0, 0.0]
input_zono = Zonotope(input_center, 0.1 * I(5))
error_zono = quantization_error_zonotope(net, quant_config, input_zono)
```
"""
function quantization_error_zonotope(net::Network, quant_config, input_zonotope::Zonotope)

    layers = net.layers
    N = length(layers)

    strat_input, bits_input, scaling_input = quant_config[:input]
    strat_weights, bits_weights, scaling_weights = quant_config[:weights]
    strat_biases, bits_biases, scaling_biases = quant_config[:biases]
    strat_activations, bits_activations, scaling_activations = quant_config[:activations]

    Z = input_zonotope
    Z_hat = quantize_zonotope(Z, scaling_input, bits_input)

    center_tilda = Z_hat.center - Z.center
    gen_tilda = genmat(Z_hat) - genmat(Z)
    Z_tilda = Zonotope(center_tilda, gen_tilda)

    factor = 2 ^ scaling_activations

    for i in 1:N

        first_factor = i == 1 ? factor : 1

        layer = layers[i]
        W, b = layer.weights, layer.bias

        W_hat = first_factor * quantize_tensor(W, scaling_weights, bits_weights)
        b_hat = factor * quantize_tensor(b, scaling_biases, bits_biases)
        W, b = first_factor * W, factor * b 

        delta_W = W_hat - W
        delta_b = b_hat - b

        center_tilda = delta_W * Z.center + W_hat * Z_tilda.center + delta_b
        gen_tilda = delta_W * genmat(Z) + W_hat * genmat(Z_tilda)
        Z_tilda = Zonotope(center_tilda, gen_tilda)

        Z = Zonotope(W * Z.center + b, W * genmat(Z))
        Z_hat = Zonotope(Z.center + Z_tilda.center, genmat(Z) + genmat(Z_tilda))

        if isa(layer.activation, ReLU)

            lambda, mu, E = abstract_relu_triplet(Z)
            lambda_hat, mu_hat, E_hat = abstract_round_clamp_triplet(Z_hat, 2 ^ bits_activations - 1)

            delta_lambda, delta_mu, delta_E = lambda_hat - lambda, mu_hat - mu, E_hat - E

            center_tilda = delta_lambda * Z.center + lambda_hat * Z_tilda.center + delta_mu
            gen_tilda = hcat(delta_lambda * genmat(Z) + lambda_hat * genmat(Z_tilda), delta_E)
            Z_tilda = Zonotope(center_tilda, gen_tilda)

            Z = Zonotope(lambda * Z.center + mu, hcat(lambda * genmat(Z), E))

            Z_hat = Zonotope(Z.center + Z_tilda.center, genmat(Z) + genmat(Z_tilda))

        end
    end

    Z_tilda = Zonotope(Z_tilda.center / factor, genmat(Z_tilda) / factor)
    return Z_tilda
end

