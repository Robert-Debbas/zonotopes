# Setup Instructions

This guide explains how to set up the dependencies and data files needed to run the benchmarks.

## Prerequisites

- Julia 1.6 or higher
- Git

## 1. Install Julia Dependencies

```bash
cd QuantizedZonotopeVerification
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs all required packages listed in `Project.toml`.

## 2. Setup Network Files

The ACAS Xu network files are not included in the repository (they're gitignored). You need to obtain them separately.

### Option A: From QEBVerif (If Available)

If you have access to the QEBVerif repository (which is also gitignored), the networks are typically in their benchmark directory. You would need to convert them to the JSON format expected by this code.

### Option B: Create Networks Directory (For Your Own Networks)

If you have ACAS Xu networks in the correct JSON format:

```bash
mkdir -p networks
# Copy your network JSON files here
cp /path/to/your/networks/*.json networks/
```

**Expected JSON format:**
```json
{
  "layer1": {
    "W": [[w11, w12, ...], [w21, w22, ...], ...],
    "b": [b1, b2, ...]
  },
  "layer2": {
    "W": [...],
    "b": [...]
  },
  ...
}
```

Where:
- `W` is the weight matrix (as array of columns)
- `b` is the bias vector
- Layers are named sequentially: "layer1", "layer2", etc.

### Option C: Use ModelVerification.jl Networks

The `ModelVerification.jl` submodule may contain network utilities. If it's set up:

```bash
git submodule update --init --recursive
```

Then check `ModelVerification.jl/` for network loading utilities.

### Option D: Skip Network Experiments

If you don't have ACAS Xu networks, you can still:
- Run unit tests: `julia --project=. test/runtests.jl`
- Test on synthetic networks (see `test/test_network.jl` for examples)
- Adapt benchmarks to use your own networks

## 3. Setup QEBVerif Baseline (Optional)

The QEBVerif baseline comparison (Python implementation) is also gitignored.

If you want to compare with QEBVerif:

1. Clone QEBVerif: https://github.com/Lawepwasd/QEBVerif
   ```bash
   git clone https://github.com/Lawepwasd/QEBVerif.git
   ```

2. Install its dependencies (see QEBVerif README)

3. Make sure it can access the same networks

## 4. Verify Setup

Test that the core functionality works:

```bash
# Run unit tests (no networks needed)
julia --project=. test/runtests.jl

# Test with synthetic network
julia --project=. test/test_network.jl
```

## 5. Run Benchmarks

Once networks are set up:

```bash
cd benchmarks
julia --project=.. run_all_benchmarks.jl
```

Or run individual benchmarks:
```bash
julia --project=.. acasxu_experiments.jl
julia --project=.. comparison_with_sampling.jl
julia --project=.. validation_vs_modelverification.jl
```

## Troubleshooting

### "No such file or directory: networks/acasxu_X_Y.json"

The benchmark scripts expect networks in `networks/acasxu_X_Y.json` format. Either:
- Add the networks with this naming scheme
- Modify the benchmark scripts to point to your network files
- Skip ACAS Xu benchmarks and test on your own networks

### Package Precompilation Warnings

The warnings about method overwriting between ReachabilityBase and LazySets can be ignored - they're due to package version conflicts but don't affect functionality.

### Missing ModelVerification.jl

If ModelVerification.jl is needed:
```bash
git submodule update --init --recursive
```

## For Collaborators

If you're reproducing results from the paper, please contact the authors for:
1. The specific ACAS Xu network JSON files used
2. The exact package versions (see `Manifest.toml`)
3. Any preprocessing scripts needed for the networks

## Network Sources

ACAS Xu networks are originally from:
- Katz et al., "Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks", CAV 2017
- Available at: https://github.com/sisl/NeuralVerification.jl/tree/master/examples/ACASXu

You may need to convert from `.nnet` format to the JSON format expected by this code.
