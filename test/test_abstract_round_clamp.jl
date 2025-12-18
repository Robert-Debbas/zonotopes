using Test
using LazySets
include("../src/QuantizedZonotopeVerification.jl")
using .QuantizedZonotopeVerification

@testset "Zonotope Abstractions" begin

    @testset "abstract_relu" begin
        Z = Zonotope([1.0, -1.0], [1.0 0.0; 0.0 1.0])
        Z_relu = abstract_relu(Z)

        # Check basic properties
        @test isa(Z_relu, Zonotope)
        @test length(Z_relu.center) == 2
        @test size(Z_relu.generators, 2) == 4  # m + n = 2 + 2

        # Example test: upper bound should be >= 0
        abs_G = sum(abs.(Z_relu.generators), dims=2)
        upper_bound = Z_relu.center + abs_G
        @test all(upper_bound .>= 0)
    end

    @testset "abstract_round_clamp" begin
        Z = Zonotope([0.5, 1.5], [0.3 0.1; 0.2 0.4])
        Z_clamped = abstract_round_clamp(Z, 2.0)

        # Basic shape tests
        @test isa(Z_clamped, Zonotope)
        @test length(Z_clamped.center) == 2
        @test size(Z_clamped.generators, 2) == 4  # m + n = 2 + 2

        # Sanity check: upper bound should be <= Cub
        abs_G = sum(abs.(Z_clamped.generators), dims=2)
        upper_bound = Z_clamped.center + abs_G
        @test all(upper_bound .<= 2.0 .+ 1e-6)  # allow small epsilon for numerical stability
    end

end
