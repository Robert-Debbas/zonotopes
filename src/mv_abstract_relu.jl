using LazySets

# Dummy types
struct Ai2z end
struct ImageZono end

# AbstractPolytope and Zonotope are from LazySets
function propagate_layer(prop_method::Union{Ai2z, ImageZono}, layer::Function, reach::AbstractPolytope, batch_info)
    reach = overapproximate(Rectification(reach), Zonotope)
    return reach
end