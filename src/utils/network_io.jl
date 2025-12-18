using JSON3
using ModelVerification

"""
    load_acasxu_network_from_json(json_path::String) -> Network

Load an ACAS Xu neural network from a JSON file.

# Arguments
- `json_path::String`: Path to the JSON file containing network weights and biases

# Returns
- `Network`: A Network object with ReLU activations for hidden layers and Identity activation for output layer

# JSON Format
The JSON file should contain layer definitions with keys for each layer (e.g., "layer1", "layer2"),
where each layer has:
- "W": Weight matrix (as array of columns)
- "b": Bias vector
"""
function load_acasxu_network_from_json(json_path::String)
    data = JSON3.read(read(json_path, String))
    layers = ModelVerification.Layer[]

    for layername in sort(collect(keys(data)))
        W_json = data[layername]["W"]
        W = Float64.(hcat(W_json...))
        b = Float64.(data[layername]["b"])
        push!(layers, ModelVerification.Layer(W, b, ModelVerification.ReLU()))
    end

    # Last layer has identity activation
    layers[end] = ModelVerification.Layer(layers[end].weights, layers[end].bias, ModelVerification.Id())
    return ModelVerification.Network(layers)
end
