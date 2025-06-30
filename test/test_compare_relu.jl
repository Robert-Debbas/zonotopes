using .QuantizedZonotopeVerification
using LazySets
using LinearAlgebra
using Printf

# Create a simple 2D zonotope
center = [1.0, -1.5]
generators = [0.5 0.1; 1 0.6]
Z = Zonotope(center, generators)

println("=== Original Zonotope ===")
println(Z)

# Run propagate_layer
relu(x) = x > 0 ? x : 0
Z_prop = propagate_layer(Ai2z(), relu, Z, nothing)

println("\n=== ModelVerification.jl ReLU ===")
println("Center: ", Z_prop.center)
println("Generators: \n", Z_prop.generators)

# Run my custom abstract_relu
Z_custom = abstract_relu(Z)

println("\n=== My abstract_relu ===")
println("Center: ", Z_custom.center)
println("Generators: \n", Z_custom.generators)

# Optional: Compare centers and generators
println("\n=== Difference (Center) ===")
println(Z_custom.center - Z_prop.center)

println("\n=== Difference (Generators) ===")
println(Z_custom.generators - Z_prop.generators)
