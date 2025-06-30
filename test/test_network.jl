using Test
using LazySets
using LinearAlgebra
using ModelVerification
include("../src/QuantizedZonotopeVerification.jl")
using .QuantizedZonotopeVerification

@testset "Quantization Error Zonotope" begin

    net = Network([
        Layer(randn(3, 2), randn(3), ReLU()),
        Layer(randn(1, 3), randn(1), Id())
    ])

    quant_config = Dict(
        :input => 3,
        :weights => 3,
        :biases => 3,
        :activations => 3,
        :range => 100
    )

    input_zono = Zonotope(zeros(2), 0.1 * Matrix(I, 2, 2))

    output_zono = quantization_error_zonotope(net, quant_config, input_zono)

    println("Output Zonotope:")
    println("Center: ", output_zono.center)
    println("Generators: ", output_zono.generators)

    box = overapproximate(output_zono, Hyperrectangle)

    println("Box lower bounds: ", box.center .- box.radius)
    println("Box upper bounds: ", box.center .+ box.radius)

end
