# Quantized Zonotope Verification

A sound and efficient method for verifying quantized neural networks using zonotope-based abstract interpretation.

## Overview

This package implements a novel approach for computing tight over-approximations of quantization error in neural networks. Unlike existing methods that rely on expensive MILP solvers, our approach uses abstract interpretation with zonotopes to propagate quantization error through the network efficiently while maintaining soundness.

### Key Features

- **Sound verification**: Provides guaranteed over-approximations of quantization error
- **Efficient computation**: Significantly faster than MILP-based methods (e.g., QEBVerif)
- **Comparable tightness**: Achieves interval tightness comparable to existing approaches like DRA
- **Flexible quantization**: Supports various bit-widths and scaling factors for inputs, weights, biases, and activations

## Algorithm

The core algorithm maintains three zonotopes per layer during forward propagation:

1. **Z**: Exact (real-valued) intermediate values
2. **Z_hat**: Quantized intermediate values
3. **Z_tilda**: Quantization error (difference between quantized and exact)

For each layer, the algorithm:
- Propagates through affine transformations using both real and quantized weights/biases
- Applies abstract ReLU or round-clamp operations
- Accumulates quantization error using custom abstract transformations

The final output is a zonotope over-approximating the quantization error bounds.

## Installation

### Prerequisites

- Julia 1.6 or higher
- Required Julia packages (see `Project.toml`):
  - LazySets
  - ModelVerification
  - JSON3
  - LinearAlgebra
  - IterTools

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Robert-Debbas/zonotopes.git
cd QuantizedZonotopeVerification

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Note:** Network files and baseline implementations are not included in the repository (gitignored). To run the ACAS Xu benchmarks, you'll need to:

1. **Obtain ACAS Xu networks**: Available from [NeuralVerification.jl](https://github.com/sisl/NeuralVerification.jl/tree/master/examples/ACASXu)
2. **Convert to JSON format**: Networks should be in `networks/acasxu_X_Y.json` format (see `src/utils/network_io.jl` for expected structure)
3. **Optional - QEBVerif baseline**: Clone from [QEBVerif repo](https://github.com/Lawepwasd/QEBVerif) for comparison

Without networks, you can still:
- Run unit tests: `julia --project=. test/runtests.jl`
- Test on synthetic networks (see `test/test_network.jl`)
- Use your own network files

## Usage

### Basic Example

```julia
using QuantizedZonotopeVerification
using LinearAlgebra

# Load a network
net = load_acasxu_network_from_json("networks/acasxu_2_5.json")

# Configure quantization parameters
quant_config = Dict(
    :input => ("pm", 8, 8),           # Plus-minus, 8 bits, 8 fractional bits
    :weights => ("pm", 8, 6),         # 8 bits total, 6 fractional bits
    :biases => ("pm", 8, 6),
    :activations => ("p", 8, 6)       # Positive only (ReLU output)
)

# Define input region as a zonotope
input_center = [0.0, 0.0, 0.0, 0.0, 0.0]
attack_radius = 0.1
input_zono = Zonotope(input_center, attack_radius * I(5))

# Compute quantization error zonotope
error_zono = quantization_error_zonotope(net, quant_config, input_zono)

# Extract interval bounds
box = overapproximate(error_zono, Hyperrectangle)
lower_bounds = box.center .- box.radius
upper_bounds = box.center .+ box.radius

println("Quantization error bounds:")
println("Lower: ", lower_bounds)
println("Upper: ", upper_bounds)
```

### Quantization Configuration

The `quant_config` dictionary specifies quantization parameters:

```julia
quant_config = Dict(
    :input => (strategy, bits, scaling),
    :weights => (strategy, bits, scaling),
    :biases => (strategy, bits, scaling),
    :activations => (strategy, bits, scaling)
)
```

- **strategy**: `"p"` (positive only: [0, 2^bits-1]) or `"pm"` (plus-minus: [-2^(bits-1), 2^(bits-1)-1])
- **bits**: Total bit-width
- **scaling**: Number of fractional bits (scaling factor = 2^scaling)

## Experiments

### Running Benchmarks

```bash
# Run all benchmarks
cd benchmarks
julia --project=.. run_all_benchmarks.jl

# Or run individual benchmarks
julia --project=.. acasxu_experiments.jl
julia --project=.. comparison_with_sampling.jl
```

### Benchmark Details

The `benchmarks/` directory contains comprehensive experiments:

- **acasxu_experiments.jl**: Main verification using zonotope-based method on ACAS Xu networks
- **comparison_with_sampling.jl**: Random sampling baseline for comparison
- **run_all_benchmarks.jl**: Run complete benchmark suite

See `benchmarks/README.md` for detailed documentation on running and customizing experiments.

### Typical Configuration

Experiments use:
- **Networks**: ACAS Xu collision avoidance networks (various sizes)
- **Quantization**: 6-8 bits with different scaling factors
- **Input regions**: Fixed points with small perturbation radii (0.01-0.1)
- **Baseline comparison**: Random sampling and QEBVerif DRA (Python, in `QEBVerif/`)

Results are saved to `results/data/` and visualizations to `results/figures/`

## Method Comparison

| Method | Type | Soundness | Efficiency | Tightness |
|--------|------|-----------|------------|-----------|
| **Ours (Zonotope)** | Abstract Interpretation | Sound (over-approx) | Fast | Comparable to DRA |
| QEBVerif DRA | Abstract Interpretation | Sound (over-approx) | Fast | Reference |
| QEBVerif MILP | Optimization | Sound & Complete | Slow (requires Gurobi) | Tight |
| Random Sampling | Monte Carlo | Unsound (under-approx) | Fast | Optimistic |

### Key Advantages

1. **No solver dependency**: Unlike QEBVerif's MILP approach, doesn't require Gurobi license
2. **Efficient**: Comparable runtime to DRA, much faster than MILP
3. **Sound**: Provides guaranteed over-approximations (unlike random sampling)
4. **Comparable tightness**: Achieves similar interval widths to DRA in most cases

## Project Structure

```
QuantizedZonotopeVerification/
├── README.md                          # This file
├── src/
│   ├── QuantizedZonotopeVerification.jl  # Main module
│   ├── core/
│   │   ├── quantized_propagation.jl   # Core algorithm (quantization_error_zonotope)
│   │   ├── abstractions.jl            # Abstract ReLU and round-clamp operations
│   │   ├── quantization.jl            # Quantization utilities
│   │   └── sampling.jl                # Sampling baseline and propagation
│   └── utils/
│       ├── conversions.jl             # Box/zonotope conversions
│       └── network_io.jl              # Network loading utilities
├── benchmarks/                        # Experimental evaluation
│   ├── acasxu_experiments.jl         # Main ACAS Xu experiments
│   ├── comparison_with_sampling.jl   # Sampling baseline comparison
│   ├── run_all_benchmarks.jl         # Run all benchmarks
│   └── README.md                     # Benchmark documentation
├── results/                           # Experimental results
│   ├── data/                         # CSV/JSON result files
│   └── figures/                      # Plots and visualizations
├── test/                             # Unit tests (if any)
├── networks/                         # ACAS Xu networks (gitignored)
├── QEBVerif/                        # QEBVerif baseline (gitignored)
└── ModelVerification.jl/            # ModelVerification.jl submodule (gitignored)
```

## Implementation Details

### Core Functions

Located in `src/core/`:

- **`quantization_error_zonotope`** (`quantized_propagation.jl`): Main verification algorithm
- **`abstract_relu_triplet`** (`abstractions.jl`): Abstract ReLU in triplet form (λ, μ, E)
- **`abstract_round_clamp_triplet`** (`abstractions.jl`): Abstract round-and-clamp for quantized activations
- **`quantize_tensor`** (`quantization.jl`): Fixed-point quantization for scalars/tensors
- **`sample_error_bounds`** (`sampling.jl`): Monte Carlo sampling baseline

### Utility Functions

Located in `src/utils/`:

- **`load_acasxu_network_from_json`** (`network_io.jl`): Load networks from JSON
- **`box_to_zonotope`, `zonotope_to_box`** (`conversions.jl`): Set conversions

### Technical Notes

- Zonotopes represented using center-generator form from LazySets.jl
- Abstract transformations decomposed into diagonal scaling + bias + error term
- Quantization uses round-to-nearest with symmetric clamping
- Networks loaded from JSON must follow ModelVerification.jl format

## Related Work

This work builds upon:

- **ModelVerification.jl**: Julia package for neural network verification
- **QEBVerif** (Zhang et al., CAV 2023): Quantization error bound verification using DRA and MILP
- **LazySets.jl**: Julia library for lazy set representations

## Citation

If you use this code, please cite:

```bibtex
@software{quantized_zonotope_verification,
  title = {Quantized Zonotope Verification},
  author = {Debbas, Robert},
  year = {2024},
  url = {https://github.com/...}
}
```

### Related Publications

- Zhang, Y., Song, F., Sun, J.: *QEBVerif: Quantization Error Bound Verification of Neural Networks*. CAV 2023.

## License

[Specify license here]

## Contact

For questions or issues, please contact [contact information].

## Acknowledgments

This work was developed with guidance from [advisors/collaborators].
