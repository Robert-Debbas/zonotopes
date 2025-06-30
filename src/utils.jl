using IterTools: product
using LazySets

function box_to_vertices(box::Vector{Tuple{Float64, Float64}})
    bounds = [[low, high] for (low, high) in box]
    verts = collect(product(bounds...))  
    return reduce(vcat, [permutedims(collect(v)) for v in verts])  
end

function vertices_to_box(vertices::Matrix{Float64})
    mins = mapslices(minimum, vertices; dims=1)
    maxs = mapslices(maximum, vertices; dims=1)
    return [(mins[i], maxs[i]) for i in 1:size(vertices, 2)]
end

function zonotope_to_box(z::Zonotope)
    hr = overapproximate(z, Hyperrectangle)
    return [(hr.center[i] - hr.radius[i], hr.center[i] + hr.radius[i]) for i in 1:length(hr.center)]
end

function quantize_tensor(x, bits, range)
    q = 2 ^ bits
    x_rounded = round.(x * q) 
    x_clipped = clamp.(x_rounded, -range, range)
    return x_clipped / q
end

function box_to_zonotope(box::Vector{Tuple{Float64, Float64}})
    d = length(box)
    center = [(low + high) / 2 for (low, high) in box]
    radii = [(high - low) / 2 for (low, high) in box]
    G = diagm(0 => radii)
    return Zonotope(center, G)
end

function quantize_zonotope(z::Zonotope, bits::Int, range::Int)
    println("\n--- quantize_zonotope ---")
    println("Original center: ", z.center)
    println("Original generators: ", z.generators)

    box = zonotope_to_box(z)
    println("Box: ", box)

    vertices = box_to_vertices(box)
    println("Vertices size: ", size(vertices))

    q_vertices = quantize_tensor(vertices, bits, range)
    println("Quantized vertices size: ", size(q_vertices))

    q_box = vertices_to_box(q_vertices)
    println("Quantized box: ", q_box)

    zq = box_to_zonotope(q_box)
    println("Quantized center: ", zq.center)
    println("Quantized generators: ", zq.generators)

    return zq
end
