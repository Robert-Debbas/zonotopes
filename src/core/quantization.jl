using LazySets

"""
    quantize_tensor(x, scaling, bits) -> Array

Quantize a tensor (scalar, vector, or matrix) using fixed-point quantization.

# Arguments
- `x`: Input tensor (scalar, vector, or matrix)
- `scaling::Int`: Number of fractional bits (scaling factor = 2^scaling)
- `bits::Int`: Total bit-width (range: [-2^(bits-1), 2^(bits-1)-1])

# Returns
- Quantized tensor of same type/shape as input

# Algorithm
1. Scale by 2^scaling
2. Round to nearest integer
3. Clamp to representable range
4. Descale by 2^scaling
"""
function quantize_tensor(x, scaling, bits)
    q = 2.0 ^ scaling
    x_rounded = round.(x * q)
    range_lower = - 2.0 ^ (bits - 1)
    range_upper = 2.0 ^ (bits - 1) - 1
    x_clipped = clamp.(x_rounded, range_lower, range_upper)
    return x_clipped / q
end

"""
    quantize_zonotope(z::Zonotope, bits::Int, range::Int) -> Zonotope

Quantize a zonotope by quantizing all its vertices and taking the bounding box.

# Arguments
- `z::Zonotope`: Input zonotope
- `bits::Int`: Number of fractional bits for quantization
- `range::Int`: Total bit-width

# Returns
- `Zonotope`: Over-approximation of the quantized zonotope as an axis-aligned zonotope
"""
function quantize_zonotope(z::Zonotope, bits::Int, range::Int)
    box = zonotope_to_box(z)
    vertices = box_to_vertices(box)
    q_vertices = quantize_tensor(vertices, bits, range)
    q_box = vertices_to_box(q_vertices)
    zq = box_to_zonotope(q_box)
    return zq
end
