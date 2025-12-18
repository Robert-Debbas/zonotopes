using IterTools: product
using LazySets
using LinearAlgebra

"""
    box_to_vertices(box::Vector{Tuple{Float64, Float64}}) -> Matrix{Float64}

Convert a box (hyperrectangle) to its vertex representation.

# Arguments
- `box::Vector{Tuple{Float64, Float64}}`: Vector of (lower, upper) bounds for each dimension

# Returns
- `Matrix{Float64}`: Matrix where each row is a vertex (2^d vertices for d dimensions)
"""
function box_to_vertices(box::Vector{Tuple{Float64, Float64}})
    bounds = [[low, high] for (low, high) in box]
    verts = collect(product(bounds...))
    return reduce(vcat, [permutedims(collect(v)) for v in verts])
end

"""
    vertices_to_box(vertices::Matrix{Float64}) -> Vector{Tuple{Float64, Float64}}

Compute the axis-aligned bounding box of a set of vertices.

# Arguments
- `vertices::Matrix{Float64}`: Matrix where each row is a vertex

# Returns
- `Vector{Tuple{Float64, Float64}}`: Vector of (lower, upper) bounds for each dimension
"""
function vertices_to_box(vertices::Matrix{Float64})
    mins = mapslices(minimum, vertices; dims=1)
    maxs = mapslices(maximum, vertices; dims=1)
    return [(mins[i], maxs[i]) for i in 1:size(vertices, 2)]
end

"""
    zonotope_to_box(z::Zonotope) -> Vector{Tuple{Float64, Float64}}

Over-approximate a zonotope with an axis-aligned bounding box.

# Arguments
- `z::Zonotope`: Input zonotope

# Returns
- `Vector{Tuple{Float64, Float64}}`: Vector of (lower, upper) bounds for each dimension
"""
function zonotope_to_box(z::Zonotope)
    hr = overapproximate(z, Hyperrectangle)
    return [(hr.center[i] - hr.radius[i], hr.center[i] + hr.radius[i]) for i in 1:length(hr.center)]
end

"""
    box_to_zonotope(box::Vector{Tuple{Float64, Float64}}) -> Zonotope

Convert an axis-aligned box to a zonotope.

# Arguments
- `box::Vector{Tuple{Float64, Float64}}`: Vector of (lower, upper) bounds

# Returns
- `Zonotope`: Zonotope with center at box midpoint and axis-aligned generators
"""
function box_to_zonotope(box::Vector{Tuple{Float64, Float64}})
    d = length(box)
    center = [(low + high) / 2 for (low, high) in box]
    radii = [(high - low) / 2 for (low, high) in box]
    G = diagm(0 => radii)
    return Zonotope(center, G)
end
