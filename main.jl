using Optim, LineSearches, Random, Distributions
using Gadfly, DataFrames
import Optim.optimize


### STRUCT AND TYPES ###
abstract type AbstractEstimator end

mutable struct MLEEstimator<:AbstractEstimator
    y::Array{Any, 1}
    θ̂::Array{Array{Float64,2},1}
    p̂::Array{Any,1}
end
mutable struct OracleEstimator<:AbstractEstimator
    y::Array{Any, 1}
    θ̂::Array{Array{Float64,2},1}
    p̂::Array{Any,1}
end
mutable struct LeastSquaresEstimator<:AbstractEstimator
    y::Array{Any, 1}
    θ̂::Array{Array{Float64,2},1}
    p̂::Array{Any,1}
end
mutable struct GMMEstimator<:AbstractEstimator
    y::Array{Any, 1}
    θ̂::Array{Array{Float64,2},1}
    p̂::Array{Any,1}
end

function MLEEstimator(θ0)
    MLEEstimator([], [θ0], [])
end
function OracleEstimator(θ0)
    OracleEstimator([], [θ0], [])
end
function LeastSquaresEstimator(θ0)
    LeastSquaresEstimator([], [θ0], [])
end
function GMMEstimator(θ0)
    GMMEstimator([], [θ0], [])
end

### END STRUCT AND TYPES ###


### UTILITIES ###
function true_utility(p, x)
    sum(hcat(p,x) .* TRUE_PARAMETERS, dims=2)
end
function true_utility(p, x, e)
    true_utility(p, x) + e
end

function choice(p, x, e)
    res = zeros(J)
    V = true_utility(p, x, e)
    if maximum(V) <= 0
        return res
    end
    res[argmax(V)] = 1
    return res
end

function linear_utility(p, x, θ)
    sum(hcat(p,x) .* θ, dims=2)
end

function multilogit(U)
    exp.(U) / (1+sum(exp.(U)))
end

function square(x)
    x*x
end
### END UTILITIES ###


### PRICE AND REVENUE ###
function revenue(p, x, θ)
    sum(p .* multilogit(linear_utility(p,x, θ)))
end

p0 = [0.2 for _ =1:J];
function best_price(θ, x)
    f = p -> -1* revenue(p, x, θ)
    optimize(f, p0, x_tol = 1e-2) |> Optim.minimizer
end

function get_best_price(estimator::OracleEstimator, X, Δp)
    best_price(estimator.θ̂[end], X)
end
function get_best_price(estimator::AbstractEstimator, X, Δp)
    bp = best_price(estimator.θ̂[end], X) + Δp
    p0 = [0.2 for _ =1:J]
    [maximum(z) for z in zip(p0, bp)]
end
### END PRICE AND REVENUE ###

### ESTIMATION ###
inner_optimizer = GradientDescent(linesearch=LineSearches.HagerZhang());
lower = [[-2.0, -0.1, -0.1] [-2.0, -0.1, -0.1] [-2.0, -0.1, -0.1]]';
upper = [[-0.2, 5.0, 5.0] [-0.2, 5.0, 5.0] [-0.2, 5.0, 5.0]]';
function optimize(estimator::AbstractEstimator, func)
    optimize(func, lower, upper, estimator.θ̂[end],
            Fminbox(inner_optimizer),
            Optim.Options(x_tol = 1e-3, time_limit = 1))
end

function estimation(estimator::MLEEstimator, X,  Δp)
    function MLE(θ)
        ll = 0
        for i=1:length(estimator.y)
            j = argmax(estimator.y[i])
            if maximum(estimator.y[i]) > 0
                ll += -log(multilogit(linear_utility(estimator.p̂[i], X[i], θ))[j])
            end
        end
        return ll
    end
    optimize(estimator, MLE) |> Optim.minimizer
end

function estimation(estimator::OracleEstimator, X,  Δp)
    estimator.θ̂[end]
end

function estimation(estimator::LeastSquaresEstimator, X,  Δp)
    function LeastSquares(θ)
        loss = 0
        for i=1:length(estimator.y)
            loss +=
                estimator.y[i] - multilogit(linear_utility(estimator.p̂[i], X[i], θ))  .|>
                square |>
                sum
        end
        return loss
    end
    optimize(estimator, LeastSquares) |> Optim.minimizer
end

function estimation(estimator::GMMEstimator, X, Δp)
    function GMM(θ)
        g = zeros(J*(M+1))
        for i=1:length(estimator.y)
            z = estimator.y[i] - multilogit(linear_utility(estimator.p̂[i], X[i], θ))
            for j=1:J
                g[j] += z[j] * Δp[i][j]
                for m=1:M
                    g[J*m + j] = z[j]*X[i][j]
                end
            end
        end
        g = g ./ length(estimator.y)
        return g' * g
    end
    optimize(estimator, GMM) |> Optim.minimizer
end

function step_period(estimator::AbstractEstimator, X, E, t )
    append!(estimator.p̂, [best_price(estimator.θ̂[t], X[t])]) #get best price
    append!(estimator.y, [choice(estimator.p̂[t], X[t], E[t])])
    append!(estimator.θ̂, [estimation(estimator, X)])
end

function step_period(estimator::AbstractEstimator, X, E, t, Δp)
    append!(estimator.p̂, [get_best_price(estimator, X[t], Δp[t])]) #get best price
    append!(estimator.y, [choice(estimator.p̂[t], X[t], E[t])])
    append!(estimator.θ̂, [estimation(estimator, X, Δp)])
end

### END ESTIMATION ###


### PARAMETERS ###


TRUE_PARAMETERS =[[-1.0, 3.2, 2.1] [-1.2, 2.7, 2.5] [-1.8, 3.5, 2.5]]';

T = 100
J = 3
M = 2

g = Gumbel()
x_dist = Uniform(0,2)

X = [rand(x_dist, J,M) for _=1:T];
E = [rand(g, J) for _=1:T ];

δ = 1.0
Δp = [];
θ0 = [[-1.2, 0.0, 0.0] [-1.2, 0.0, 0.0] [-1.2, 0.0, 0.0]]';
### END PARAMETERS ###




### PROCEDURE ###
#mle_est = MLEEstimator(θ0);
lsquares = LeastSquaresEstimator(θ0);
gmm = GMMEstimator(θ0);
oracle = OracleEstimator(TRUE_PARAMETERS);

estimators = [lsquares, oracle, gmm];


@time for t = 1:T
    append!(Δp, [[ rand()> 0.5 ? δ : -δ for _=1:J]])
    Threads.@threads for e in estimators
        step_period(e, X[1:t], E[1:t], t, Δp[1:t])
    end
    if t%50 == 0
        println("Done with step $(t)")
    end
end




### END PROCEDURE ###


### ANALYSIS AND PLOTS ###

function to_dataframe(estimator::AbstractEstimator)
    α = []
    β = []
    γ = []
    price = []
    y = []
    product = []
    period = []
    revenue = []
    cumrevenue = []
    for t=1:T
        for j=1:J
            append!(α, estimator.θ̂[t][j,3])
            append!(β, estimator.θ̂[t][j,1])
            append!(γ, estimator.θ̂[t][j,2])
            append!(price, estimator.p̂[t][j])
            append!(y, estimator.y[t][j])
            append!(product, j)
            append!(period, t)
            append!(revenue, estimator.p̂[t][j] *  estimator.y[t][j])
        end
    end
    df = DataFrame(α = α, β = β, γ = γ, price = price, y = y,
    product = categorical(product),period = period, revenue=revenue)
end

df = to_dataframe(gmm);
df2 = to_dataframe(lsquares);
or = to_dataframe(oracle);
p1 = plot(df, y="β", x="period", color="product", Geom.line)
p1 = plot(df2, y="β", x="period", color="product", Geom.line)
p2 = plot(df, x="price", color="product", Geom.density)

p2 = plot(or, x="price", color="product", Geom.density)

p = best_price(mle_est.θ̂[end], X[end])

revenue(p, X[end], mle_est.θ̂[end])
multilogit(linear_utility(p, X[end], mle_est.θ̂[end]))

mle_est.θ̂[end]

### END ANALYSIS AND PLOTS ###
