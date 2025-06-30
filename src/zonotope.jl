using LazySets
using LinearAlgebra

function abstract_relu(Z::Zonotope)
    c = Z.center
    G = Z.generators

    n = length(c)
    m = size(G, 2)

    new_c = similar(c)
    new_G = zeros(n, m + n)  

    for i in 1:n
        sum_abs = sum(abs.(G[i, :]))

        l = c[i] - sum_abs
        u = c[i] + sum_abs

        if u <= 0
            new_c[i] = 0.0
            new_G[i, :] .= 0.0

        elseif l >= 0
            new_c[i] = c[i]
            new_G[i, 1:m] = G[i, :]

        else
            a = u / (u - l)
            b = -u * l / (u - l)

            new_c[i] = a * c[i] + b / 2
            new_G[i, 1:m] = a * G[i, :]
            new_G[i, m+i] = b / 2
        end
    end

    return Zonotope(new_c, new_G)
end

function abstract_relu_triplet(Z::Zonotope)
    c = Z.center
    G = Z.generators

    n = length(c)
    m = size(G, 2)

    lambda = zeros(Float64, n, n)
    mu = zeros(Float64, n)
    E = zeros(Float64, n, n)

    for i in 1:n
        sum_abs = sum(abs.(G[i, :]))

        l = c[i] - sum_abs
        u = c[i] + sum_abs

        if u <= 0
            lambda[i, i] = 0.0
            mu[i] = 0.0

        elseif l >= 0
            lambda[i, i] = 1.0
            mu[i] = 0.0

        else
            a = u / (u - l)
            b = -u * l / (u - l)

            lambda[i, i] = a
            mu[i] = b / 2
            E[i, i] = b / 2
        end
    end

    return lambda, mu, E
end

function abstract_round_clamp(Z::Zonotope{Float64}, Cub::Float64)
    c = Z.center
    G = Z.generators

    n = length(c)
    m = size(G, 2)

    new_c = similar(c)
    new_G = zeros(Float64, n, m + n)

    for i in 1:n
        sum_abs = sum(abs.(G[i, :]))
        l = c[i] - sum_abs
        u = c[i] + sum_abs

        if u <= 0
            new_c[i] = 0.0

        elseif l >= Cub
            new_c[i] = Cub

        elseif 0 <= l && u <= Cub
            new_c[i] = c[i]
            new_G[i, 1:m] .= G[i, :]
            new_G[i, m + i] = 0.5

        elseif l <= 0 && Cub <= u

            if (Cub - l <= u && Cub - l >= 0.5) || (u <= Cub - l && l >= -0.5)
                a = Cub / (u - 0.5)
            else
                a = Cub / (Cub - l - 0.5)
            end

            b1 = max(-a * l, Cub - a * (Cub - 0.5))
            b2 = min(-0.5 * a, Cub - a * u)

            new_c[i] = a * c[i] + (b1 + b2) / 2
            new_G[i, 1:m] .= a * G[i, :]
            new_G[i, m + i] = (b1 - b2) / 2

        elseif l <= 0 && u <= Cub
            ru = round(u)

            a = ru / (ru - l - 0.5)
            b1 = -ru * l / (ru - 0.5 - l)
            b2 = 0.5 * ru / (ru - l - 0.5)

            new_c[i] = a * c[i] + (b1 + b2) / 2
            new_G[i, 1:m] .= a * G[i, :]
            new_G[i, m + i] = (b1 - b2) / 2

        else
            rl = round(l)

            a = (Cub - rl) / (u - rl - 0.5)
            b1 = Cub - a * (Cub - 0.5)
            b2 = Cub - a * u

            new_c[i] = a * c[i] + (b1 + b2) / 2
            new_G[i, 1:m] .= a * G[i, :]
            new_G[i, m + i] = (b1 - b2) / 2
        end
    end

    return Zonotope(new_c, new_G)
end

function abstract_round_clamp_triplet(Z::Zonotope{Float64}, Cub::Int64)
    c = Z.center
    G = Z.generators

    n = length(c)
    m = size(G, 2)

    lambda = zeros(Float64, n, n)
    mu = zeros(Float64, n)
    E = zeros(Float64, n, n)

    for i in 1:n
        sum_abs = sum(abs.(G[i, :]))
        l = c[i] - sum_abs
        u = c[i] + sum_abs

        if u <= 0
            lambda[i, i] = 0.0
            mu[i] = 0.0

        elseif l >= Cub
            lambda[i, i] = 0.0
            mu[i] = Cub

        elseif 0 <= l && u <= Cub
            lambda[i, i] = 1.0
            mu[i] = 0.0
            E[i, i] = 0.5

        elseif l <= 0 && Cub <= u
            if (Cub - l <= u && Cub - l >= 0.5) || (u <= Cub - l && l >= -0.5)
                a = Cub / (u - 0.5)
            else
                a = Cub / (Cub - l - 0.5)
            end
            b1 = max(-a * l, Cub - a * (Cub - 0.5))
            b2 = min(-0.5 * a, Cub - a * u)

            lambda[i, i] = a
            mu[i] = (b1 + b2) / 2
            E[i, i] = (b1 - b2) / 2

        elseif l <= 0 && u <= Cub
            ru = round(u)

            a = ru / (ru - l - 0.5)
            b1 = -ru * l / (ru - 0.5 - l)
            b2 = 0.5 * ru / (ru - l - 0.5)

            lambda[i, i] = a
            mu[i] = (b1 + b2) / 2
            E[i, i] = (b1 - b2) / 2

        else
            rl = round(l)

            a = (Cub - rl) / (u - rl - 0.5)
            b1 = Cub - a * (Cub - 0.5)
            b2 = Cub - a * u

            lambda[i, i] = a
            mu[i] = (b1 + b2) / 2
            E[i, i] = (b1 - b2) / 2
        end
    end

    return lambda, mu, E
end