using Test
using LazySets
using LinearAlgebra
using ModelVerification
include("../src/QuantizedZonotopeVerification.jl")
using .QuantizedZonotopeVerification

net_nb1 = 2
net_nb2 = 7
input_nb = 4
attack_rad = 0.1
Q = 6

# Load network
net = load_acasxu_network_from_json("networks/acasxu_$(net_nb1)_$(net_nb2).json")

quant_config = Dict(
    :input => ("pm", 8, 8),
    :weights => ("pm", Q, Q - 2),
    :biases => ("pm", Q, Q - 2),
    :activations => ("p", Q, Q - 2)
)

# Define input center
x_input_real_map = Dict(
    1 => [0.0, 0.0, 0.0, 0.0, 0.0],
    2 => [0.2, -0.1, 0.0, -0.3, 0.4],
    3 => [0.45, -0.23, -0.4, 0.12, 0.33],
    4 => [-0.2, -0.25, -0.5, -0.3, -0.44],
    5 => [0.61, 0.36, 0.0, 0.0, -0.24]
)

input = x_input_real_map[input_nb]

# Run sampling
error_min, error_max = sample_error_bounds(net, quant_config, input, attack_rad, num_samples=1000)

target_cls_map = Dict(1 => 1, 2 => 1, 3 => 1, 4 => 1, 5 => 1)
original_prediction = target_cls_map[input_nb]

println("Error interval:")
println("Lower bounds: ", error_min[original_prediction])
println("Upper bounds: ", error_max[original_prediction])
