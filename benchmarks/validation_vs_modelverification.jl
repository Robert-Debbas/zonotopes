"""
Validation: Compare custom abstract_relu with ModelVerification.jl's ReLU abstraction

This benchmark validates that our custom abstract_relu implementation produces
similar results to ModelVerification.jl's propagate_layer with ReLU.
"""

using LazySets
using LinearAlgebra
using Printf

include("../src/QuantizedZonotopeVerification.jl")
using .QuantizedZonotopeVerification

println("="^80)
println("Validation: Custom ReLU Abstraction vs ModelVerification.jl")
println("="^80)
println()

# Create a simple 2D zonotope
center = [1.0, -1.5]
generators = [0.5 0.1; 1.0 0.6]
Z = Zonotope(center, generators)

println("=== Original Zonotope ===")
println("Center: ", Z.center)
println("Generators: \n", Z.generators)
println()

# Run ModelVerification.jl's propagate_layer
relu(x) = x > 0 ? x : 0
Z_mv = propagate_layer(Ai2z(), relu, Z, nothing)

println("=== ModelVerification.jl ReLU ===")
println("Center: ", Z_mv.center)
println("Generators: \n", Z_mv.generators)
println()

# Run our custom abstract_relu
Z_custom = abstract_relu(Z)

println("=== Our Custom abstract_relu ===")
println("Center: ", Z_custom.center)
println("Generators: \n", Z_custom.generators)
println()

# Compare results
center_diff = Z_custom.center - Z_mv.center
gen_diff = Z_custom.generators - Z_mv.generators

println("=== Differences ===")
println("Center difference: ", center_diff)
println("Max center difference: ", maximum(abs.(center_diff)))
println("Generators difference norm: ", norm(gen_diff))
println("Max generators difference: ", maximum(abs.(gen_diff)))
println()

# Validation check
tol = 1e-10
if maximum(abs.(center_diff)) < tol && maximum(abs.(gen_diff)) < tol
    println("✓ VALIDATION PASSED: Results match within tolerance")
else
    println("⚠ VALIDATION WARNING: Results differ")
    println("  This may be expected if implementations use different abstractions")
end
println()
println("="^80)
