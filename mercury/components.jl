## -- Activation functions -- ##

struct ReLU
end;
forward(x::Array{Float64, 2}, act_fn::ReLU)::Array{Float64,2} = x.*(x.>0)
gradient(x::Array{Float64, 2}, act_fn::ReLU)::Array{Float64,2} = Array{Float64, 2}(x.>0)


struct Sigmoid
end;
forward(x::Array{Float64, 2}, act_fn::Sigmoid)::Array{Float64,2} = 1 ./ (1 .+ exp.(-x))
gradient(x::Array{Float64, 2}, act_fn::Sigmoid)::Array{Float64,2} = 1 ./ (1 .+ exp.(-x)) .* (1 .- 1 ./ (1 .+ exp.(-x)))


struct Softmax
    #forward method requires inputs as probabilities
    #backward method requires inputs as probabilities
end;
forward(x::Array{Float64, 2}, act_fn::Softmax)::Array{Float64,2} = softmax(x)
gradient(x::Array{Float64, 2}, act_fn::Softmax)::Array{Float64,2} = (softmax(x)) .* (1 .- softmax(x))

function softmax(x::Array{Float64,2})::Array{Float64,2}
    #converts real numbers to probabilities
    c=maximum(x)
    p = x .- log.( sum( exp.(x .- c) ) ) .-c
    p = exp.(p)
    return p
end


## -- Loss functions -- ##

# MSE losses make no assumptions about the distribution of the inputs.
mse_loss(y_hat::Array{Float64,2}, y::Array{Float64,2}) = (y_hat - y)'*(y_hat-y)
mse_loss_derivative(y_hat::Array{Float64,2}, y::Array{Float64,2}) = (y - y_hat) ./ size(y, 1)

# xe_losses assume inputs as probabilities (real valued and normalised, ie. after a Softmax).
xe_loss(y_hat::Array{Float64,2}, y::Array{Float64,2}) = -sum(y.*log.(y_hat))
xe_loss_derivative(y_hat::Array{Float64,2}, y::Array{Float64,2}) = y_hat - y


## -- Neural network components -- ##

mutable struct Dense
    # Fully connected layer. By default has weights, biases, and an activation function.
    weight
    bias
    act_fn
    Dense(dim_in, dim_out, act_fn)=new(kaiming(Float64, dim_out, dim_in), zeros(Float64, dim_out, 1), act_fn)
end;

# Weight initialisation
function kaiming(type, dim_out,dim_in)
    # Weight init for a layer with ReLU activation function.
    matrix = randn(type, dim_out, dim_in).*sqrt(2/dim_in)
    return matrix
end
