using Test

@testset "QuantizedZonotopeVerification Tests" begin

    @testset "Abstraction Functions" begin
        include("test_abstract_round_clamp.jl")
    end

    @testset "Network Quantization" begin
        include("test_network.jl")
    end

end
