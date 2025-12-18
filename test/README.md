# Unit Tests

This directory contains **unit tests** for verifying the correctness of individual components.

## Test vs Benchmarks

**Tests (this directory):**
- Unit tests for individual functions
- Correctness validation on simple/synthetic examples
- Fast to run
- Run via `julia --project=.. test/runtests.jl`

**Benchmarks (../benchmarks/):**
- End-to-end experiments on real networks
- Performance evaluation
- Comparison with baselines
- Generate data for publication

## Available Tests

### `test_abstract_round_clamp.jl`
Unit tests for abstraction functions:
- `abstract_relu`: Tests ReLU abstraction on simple zonotopes
- `abstract_round_clamp`: Tests round-and-clamp abstraction

**What it tests:**
- Output types and shapes are correct
- Basic soundness properties (e.g., ReLU output bounds are non-negative)
- Numerical stability

### `test_network.jl`
Tests the main quantization error propagation algorithm on a small synthetic network.

**What it tests:**
- `quantization_error_zonotope` runs without errors
- Output zonotope has correct shape
- Box over-approximation works

## Running Tests

```bash
# Run all unit tests
cd test
julia --project=.. runtests.jl

# Run specific test file
julia --project=.. test_abstract_round_clamp.jl
julia --project=.. test_network.jl
```

## Adding New Tests

When adding new functionality:

1. **Add unit test here** if:
   - Testing a single function/component
   - Using synthetic/simple examples
   - Validating correctness properties

2. **Add to benchmarks/** if:
   - Testing end-to-end on real networks
   - Measuring performance
   - Comparing with baselines
   - Generating publication data

## Test Coverage

Current test coverage:
- ✓ Abstract ReLU
- ✓ Abstract round-and-clamp
- ✓ Quantization error propagation (basic)
- ⚠ Missing: Quantization utilities (quantize_tensor, quantize_zonotope)
- ⚠ Missing: Conversion functions (box_to_zonotope, etc.)
- ⚠ Missing: Sampling functions

**Note:** Some functions are implicitly tested through integration tests in `test_network.jl`.
