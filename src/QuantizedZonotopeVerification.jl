module QuantizedZonotopeVerification

include("utils.jl")
include("zonotope.jl")
include("layer.jl")
include("mv_abstract_relu.jl")
include("network.jl")

export quantize_zonotope,
       quantization_error_zonotope,
       box_to_zonotope,
       quantize_tensor,
       zonotope_to_box,
       vertices_to_box,
       box_to_vertices,
       Zonotope,
       Layer,
       Network,
       ReLU,
       Id,
       abstract_relu_triplet,
       abstract_round_clamp_triplet
end