using Test
using LazySets
using LinearAlgebra
using ModelVerification
include("../src/QuantizedZonotopeVerification.jl")
using .QuantizedZonotopeVerification

@testset "Quantization Error Zonotope" begin

    # net = Network([
    #     Layer(randn(3, 2), randn(3), ReLU()),
    #     Layer(randn(1, 3), randn(1), Id())
    # ])

    net = Network([
    # First layer: 3 neurons, input dim = 2
    Layer(
        [ 1.0   0.5;   # neuron 1
         -0.3   0.8;   # neuron 2
          0.7  -0.6],  # neuron 3
        [0.1, -0.2, 0.3],   # biases (length 3)
        ReLU()
        ),

    # Second layer: 1 neuron, input dim = 3
    Layer(
        [0.4  -0.7  1.2],   # weights (1×3)
        [0.5],              # bias (length 1)
        Id()
        )   
    ])

    # Q in {4, 6, 8, 10}
    net_nb1 = 1
    net_nb2 = 4
    input_nb = 4
    attack_rad = 0.01
    Q = 10

    quant_config = Dict(
    :input => ("pm", 8, 8),
    :weights => ("pm", Q, Q - 2),
    :biases => ("pm", Q, Q - 2),
    :activations => ("p", Q, Q - 2)
    )

    input_zono = Zonotope(zeros(2), attack_rad * Matrix(I, 2, 2))

    output_zono = quantization_error_zonotope(net, quant_config, input_zono)

    println("Output Zonotope:")
    println("Center: ", output_zono.center)
    println("Generators: ", output_zono.generators)

    box = overapproximate(output_zono, Hyperrectangle)

    println("Box lower bounds: ", box.center .- box.radius)
    println("Box upper bounds: ", box.center .+ box.radius)

end
