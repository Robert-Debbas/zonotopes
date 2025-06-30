using ModelVerification
import ModelVerification: ReLU, Id, Network, Layer
using LazySets
using LinearAlgebra
using .QuantizedZonotopeVerification

function quantization_error_zonotope(net::Network, quant_config, input_zonotope::Zonotope)

    layers = net.layers
    N = length(layers)

    bit_input = quant_config[:input]
    bit_weights = quant_config[:weights]
    bit_biases = quant_config[:biases]
    quant_range = quant_config[:range]

    Z = input_zonotope
    Z_hat = quantize_zonotope(Z, bit_input, quant_range)

    center_tilda = Z_hat.center - Z.center
    gen_tilda = genmat(Z_hat) - genmat(Z)
    Z_tilda = Zonotope(center_tilda, gen_tilda)

    for i in 1:N
        layer = layers[i]
        W, b = layer.weights, layer.bias

        W_hat = quantize_tensor(W, bit_weights, quant_range)
        b_hat = quantize_tensor(b, bit_biases, quant_range)

        delta_W = W_hat - W
        delta_b = b_hat - b

        center_tilda = delta_W * Z.center + W_hat * Z_tilda.center + delta_b
        gen_tilda = delta_W * genmat(Z) + W_hat * genmat(Z_tilda)
        Z_tilda = Zonotope(center_tilda, gen_tilda)

        Z = Zonotope(W * Z.center + b, W * genmat(Z))
        Z_hat = Zonotope(Z.center + Z_tilda.center, genmat(Z) + genmat(Z_tilda))

        if isa(layer.activation, ReLU)

            lambda, mu, E = abstract_relu_triplet(Z)
            lambda_hat, mu_hat, E_hat = abstract_round_clamp_triplet(Z_hat, quant_range)

            delta_lambda, delta_mu, delta_E = lambda_hat - lambda, mu_hat - mu, E_hat - E

            center_tilda = delta_lambda * Z.center + lambda_hat * Z_tilda.center + delta_mu
            gen_tilda = hcat(delta_lambda * genmat(Z) + lambda_hat * genmat(Z_tilda), delta_E)
            Z_tilda = Zonotope(center_tilda, gen_tilda)

            Z = Zonotope(lambda * Z.center + mu, hcat(lambda * genmat(Z), E))

            Z_hat = Zonotope(Z.center + Z_tilda.center, genmat(Z) + genmat(Z_tilda))

        end
    end

    return Z_tilda
end

