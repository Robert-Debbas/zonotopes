"""
Run all benchmarks for QuantizedZonotopeVerification

This script runs comprehensive experiments comparing:
1. Zonotope-based quantization error propagation (our method)
2. Random sampling baseline

Results are saved to the results/ directory.
"""

using Dates

println("="^80)
println("QuantizedZonotopeVerification Benchmark Suite")
println("Started at: ", Dates.now())
println("="^80)
println()

# Track timing
start_time = time()

# Run validation
println("Running validation benchmark...")
println("-"^80)
include("validation_vs_modelverification.jl")
println()

# Run ACAS Xu experiments with zonotope method
println("Running ACAS Xu experiments with zonotope method...")
println("-"^80)
include("acasxu_experiments.jl")
println()

# Run sampling comparison
println("Running sampling baseline comparison...")
println("-"^80)
include("comparison_with_sampling.jl")
println()

# Summary
elapsed = time() - start_time
println("="^80)
println("All benchmarks completed!")
println("Total time: ", round(elapsed, digits=2), " seconds")
println("Finished at: ", Dates.now())
println("="^80)
