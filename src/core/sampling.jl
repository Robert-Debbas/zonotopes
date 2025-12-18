using LinearAlgebra
using Random
using ModelVerification
import ModelVerification: ReLU, Id, Network, Layer

"""
    propagate(net::Network, x::Vector{Float64}) -> Vector{Float64}

Propagate an input through the network with real-valued (non-quantized) computation.

# Arguments
- `net::Network`: Neural network
- `x::Vector{Float64}`: Input vector

# Returns
- `Vector{Float64}`: Network output
"""
function propagate(net::Network, x::Vector{Float64})
    z = x
    for (i, layer) in enumerate(net.layers)
        z = layer.weights * z .+ layer.bias

        if isa(layer.activation, ReLU)
            z = max.(z, 0.0)

        elseif isa(layer.activation, Id)
        else
            error("Unsupported activation in propagate")
        end
    end
    return z
end

"""
    propagate_quantized(net::Network, quant_config, x::Vector{Float64}) -> Vector{Float64}

Propagate an input through the network with quantized computation.

Simulates the forward pass as it would occur on quantized hardware, with quantization
applied to inputs, weights, biases, and activations.

# Arguments
- `net::Network`: Neural network
- `quant_config::Dict`: Quantization configuration (same format as quantization_error_zonotope)
- `x::Vector{Float64}`: Input vector

# Returns
- `Vector{Float64}`: Quantized network output
"""
function propagate_quantized(net::Network, quant_config, x::Vector{Float64})
    strat_input, bits_input, scaling_input = quant_config[:input]
    strat_weights, bits_weights, scaling_weights = quant_config[:weights]
    strat_biases, bits_biases, scaling_biases = quant_config[:biases]
    strat_activations, bits_activations, scaling_activations = quant_config[:activations]

    xq = quantize_tensor(x, scaling_input, bits_input)
    z = xq
    factor = 2.0 ^ scaling_activations

    for (i, layer) in enumerate(net.layers)

        first_factor = i == 1 ? factor : 1.0
        
        Wq = quantize_tensor(layer.weights, scaling_weights, bits_weights)
        bq = quantize_tensor(layer.bias, scaling_biases, bits_biases)

        z = first_factor * Wq * z .+ factor * bq

        if isa(layer.activation, ReLU)
            z = clamp.(z, 0.0, (2.0 ^ bits_activations) - 1)
        elseif isa(layer.activation, Id)
        else
            error("Unsupported activation type in quantized pass")
        end
    end
    return z / factor
end

"""
    sample_error_bounds(net::Network, quant_config::Dict, input::Vector{Float64},
                       radius::Float64; num_samples::Int=2) -> (Vector{Float64}, Vector{Float64})

Estimate quantization error bounds via random sampling (baseline method).

Samples random points in the input region, computes both real-valued and quantized outputs,
and returns the min/max quantization error observed across all samples.

# Arguments
- `net::Network`: Neural network
- `quant_config::Dict`: Quantization configuration
- `input::Vector{Float64}`: Center of input region
- `radius::Float64`: Radius of input region (uniform in all dimensions)
- `num_samples::Int=2`: Number of random samples to draw

# Returns
- `mins::Vector{Float64}`: Lower bounds on quantization error per output dimension
- `maxs::Vector{Float64}`: Upper bounds on quantization error per output dimension

# Note
This provides sound under-approximations: true error bounds may be wider.
Used as a baseline for comparison with the sound zonotope-based method.
"""
function sample_error_bounds(net::Network, quant_config::Dict, input::Vector{Float64}, radius::Float64; num_samples::Int=2)
    d = length(input)
    println("Input dimension: $d")
    println("Input center: ", input)
    println("Sampling radius: ", radius)
    println("Number of samples: ", num_samples)

    output_dim = length(net.layers[end].bias)
    println("Network output dimension: ", output_dim)

    diffs = zeros(num_samples, output_dim)

    for i in 1:num_samples
        point = input .+ radius .* (2 .* rand(d) .- 1)
        y = propagate(net, point)
        y_q = propagate_quantized(net, quant_config, point)
        diff = y_q .- y

        diffs[i, :] .= diff

    end

    mins = vec(minimum(diffs, dims=1))
    maxs = vec(maximum(diffs, dims=1))

    return mins, maxs
end

