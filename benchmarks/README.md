# Benchmarks

This directory contains benchmark scripts for evaluating the quantized zonotope verification method.

## Prerequisites

**Note:** These benchmarks require ACAS Xu network files which are not included in the repository. See **[../SETUP.md](../SETUP.md)** for instructions on obtaining and setting up network files.

To run without networks, you can:
- Run unit tests instead: `julia --project=.. ../test/runtests.jl`
- Use the validation benchmark: `julia --project=.. validation_vs_modelverification.jl`
- Adapt scripts to use your own networks

## Available Benchmarks

### 1. `acasxu_experiments.jl`
Main experiments using the zonotope-based quantization error propagation on ACAS Xu networks.

**Configuration:**
- Network: ACAS Xu 2-5
- Quantization: 8-bit with configurable scaling
- Input: Fixed point #4 with perturbation radius 0.1
- Output: Quantization error bounds for the predicted class

**Run:**
```bash
julia --project=.. acasxu_experiments.jl
```

### 2. `comparison_with_sampling.jl`
Random sampling baseline for comparison. Provides empirical (unsound) error bounds.

**Configuration:**
- Network: ACAS Xu 2-7
- Quantization: 6-bit with configurable scaling
- Samples: 1000 random points in the input region
- Output: Min/max observed quantization errors

**Run:**
```bash
julia --project=.. comparison_with_sampling.jl
```

### 3. `validation_vs_modelverification.jl`
Validation benchmark comparing our custom abstract ReLU with ModelVerification.jl's implementation.

**Purpose:**
- Validates correctness of custom abstractions
- Compares results with established ModelVerification.jl library
- Useful for debugging and verification

**Run:**
```bash
julia --project=.. validation_vs_modelverification.jl
```

### 4. `run_all_benchmarks.jl`
Convenience script to run all benchmarks sequentially.

**Run:**
```bash
julia --project=.. run_all_benchmarks.jl
```

## Benchmark Configuration

### ACAS Xu Fixed Points
Standard test inputs from QEBVerif paper:
```julia
1: [0.0, 0.0, 0.0, 0.0, 0.0]
2: [0.2, -0.1, 0.0, -0.3, 0.4]
3: [0.45, -0.23, -0.4, 0.12, 0.33]
4: [-0.2, -0.25, -0.5, -0.3, -0.44]
5: [0.61, 0.36, 0.0, 0.0, -0.24]
```

### Quantization Settings
```julia
quant_config = Dict(
    :input => ("pm", 8, 8),        # Plus-minus, 8 bits, 8 fractional
    :weights => ("pm", Q, Q-2),    # Q bits, Q-2 fractional
    :biases => ("pm", Q, Q-2),
    :activations => ("p", Q, Q-2)  # Positive only (post-ReLU)
)
```

Common values for Q: 4, 6, 8, 10

### Attack Radii
Typical values: 0.01, 0.05, 0.1 (around input fixed point)

## Customizing Benchmarks

To test different configurations, modify the parameters:

```julia
# Network selection
net_nb1 = 2  # First index (1-5)
net_nb2 = 5  # Second index (1-9)

# Input point
input_nb = 4  # Fixed point number (1-5)

# Perturbation
attack_rad = 0.1  # Radius for input region

# Quantization bits
Q = 8  # Bit-width
```

## Expected Output

### Zonotope Method
```
Bounds for output neuron corresponding to original prediction (class 0):
Lower bound: -0.0234
Upper bound: 0.0187
```

### Sampling Baseline
```
Error interval:
Lower bounds: -0.0156
Upper bounds: 0.0142
```

**Note:** Sampling typically gives tighter (but unsound) bounds since it only observes a finite number of points.

## Adding New Benchmarks

To add a new benchmark:

1. Create a new `.jl` file in this directory
2. Include the main module: `include("../src/QuantizedZonotopeVerification.jl")`
3. Load networks, configure quantization, run experiments
4. Optionally save results to `../results/data/`
5. Add to `run_all_benchmarks.jl` if desired

## Saving Results

Results can be saved programmatically:

```julia
using CSV, DataFrames

results = DataFrame(
    network = ["acasxu_2_5"],
    bits = [8],
    lower_bound = [lower_bounds[1]],
    upper_bound = [upper_bounds[1]]
)

CSV.write("../results/data/experiment_results.csv", results)
```
