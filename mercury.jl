#%%
struct ReLU
end;
forward(x::Array{Float64, 2}, act_fn::ReLU)::Array{Float64,2} = x.*(x.>0)
gradient(x::Array{Float64, 2}, act_fn::ReLU)::Array{Float64,2} = Array{Float64, 2}(x.>0)


struct Sigmoid
end;
forward(x::Array{Float64, 2}, act_fn::Sigmoid)::Array{Float64,2} = 1 ./ (1 .+ exp.(-x))
gradient(x::Array{Float64, 2}, act_fn::Sigmoid)::Array{Float64,2} = 1 ./ (1 .+ exp.(-x)) .* (1 .- 1 ./ (1 .+ exp.(-x)))


function softmax(x::Array{Float64,2})::Array{Float64,2}
    c=maximum(x)
    out = x .- log.( sum( exp.(x .- c) ) ) .-c
    out = exp.(out)
    return out
end

struct Softmax
end;
forward(x::Array{Float64, 2}, act_fn::Softmax)::Array{Float64,2} = softmax(x)
gradient(x::Array{Float64, 2}, act_fn::Softmax)::Array{Float64,2} = (softmax(x)) .* (1 .- softmax(x))


function kaiming(type, dim_out,dim_in)
    matrix = randn(type, dim_out, dim_in).*sqrt(2/dim_in)
    return matrix
end

mutable struct Dense
    weight
    bias
    act_fn
    Dense(dim_in, dim_out, act_fn)=new(kaiming(Float64, dim_out, dim_in), zeros(Float64, dim_out), act_fn)
end;


function forward(x::Array{Float64,2}, net::Dict)::Array{Float64,2}
    A=[x]
    Z=[]
    for n in 1:length(net["Layers"])
        z = net["Layers"][n].weight*x + net["Layers"][n].bias
        x = forward(z, net["Layers"][n].act_fn)
        append!(Z, [z])
        append!(A, [x])
    end
    net["A"]=A
    net["Z"]=Z
    return x
end;


#%%
using LinearAlgebra

mse_loss(y_hat::Array{Float64,2}, y::Array{Float64,2}) = (y_hat - y)'*(y_hat-y)
mse_loss_derivative(y_hat::Array{Float64,2}, y::Array{Float64,2}) = (y - y_hat) ./ size(y, 1)

xe_loss(y_hat::Array{Float64,2}, y::Array{Float64,2}) = -sum(y.*log.(y_hat))
xe_loss_derivative(y_hat::Array{Float64,2}, y::Array{Float64,2}) = y_hat - y

function layer_backward_pass(dA, W, B, Z, A_prev, act_fn)
    m = size(A_prev, 1)
    dZ = dA.*gradient(Z, act_fn)
    dW = (dZ * A_prev') ./ m
    dB = dZ ./ m
    dA_prev = W'*dZ
    out=[dA_prev, dW, dB]
    return out
end


function full_backward_pass(grad, net)
    m = size(grad, 1)
    dA_prev = grad
    depth=length(net["Layers"])

    dW=[]
    dB=[]

    for n in 1:depth
        n_curr = depth-(n-1);
        n_prev = depth-n;
        
        dA = dA_prev;
        Z = net["Z"][n_curr];
        A_prev = net["A"][n_curr];

        W = net["Layers"][n_curr].weight;
        B = net["Layers"][n_curr].bias;
        act_fn = net["Layers"][n_curr].act_fn;

        out = layer_backward_pass(dA, W, B, Z, A_prev, act_fn)     
        
        dA_prev = out[1]
        append!(dW, [out[2]])
        append!(dB, [out[3]])   
    end

    gradients=[reverse(dW), reverse(dB)]
    return gradients

end;

function update(gradients, learning_rate, net)
    for n in 1:length(net["Layers"])
        W = net["Layers"][n].weight-=learning_rate*gradients[1][n];
        B = net["Layers"][n].bias-=learning_rate*gradients[2][n];
    end
end;

#%% Initialise the net
# [[layer1_inputdim, layer1_outputdim] [layer2_inputdim, layer2_outputdim] [layer3_inputdim, layer3_outputdim] ... ]
net=Dict("Layers"=>[], "A"=>[], "Z"=>[])
dims=[[28^2,20^2] [20^2, 20^2] [20^2, 20^2] [20^2, 16^2] [16^2,100] [100,64] [64,10]]
layers=[]
for i in 1:size(dims,2)-1
    append!(layers, [Dense(dims[1,i], dims[2,i], ReLU())])
end

head=[Dense(dims[1, size(dims,2)], dims[2, size(dims,2)], Softmax())]
append!(layers, head);
net["Layers"]=layers;


#%% Initialiset the training data
using MLDatasets
train_x, train_y = MNIST.traindata(Float64);
test_x, test_y = MNIST.testdata(Float64);

train_y_one_hot=zeros(Float64, 10, size(train_y, 1));
for i in 1:size(train_y, 1)
    label=train_y[i]+1
    train_y_one_hot[label, i]=1
end;

test_y_one_hot=zeros(Float64, 10, size(test_y, 1));
for i in 1:size(test_y, 1)
    label=test_y[i]+1
    test_y_one_hot[label, i]=1
end;

#%% Train
mb_size=100
avg_epoch_train_loss=[]
avg_epoch_test_loss=[]
avg_epoch_train_acc=[]
avg_epoch_test_acc=[]

for epoch in 1:50

    epoch_train_loss=[]
    epoch_test_loss=[]
    epoch_train_correct=0.0
    epoch_test_correct=0.0

    mb_counter=1;
    loss_grad=zeros(Float64,10,1);
    loss=0.0;
    
    println("Running epoch ", epoch)
    for i in 1:size(train_y,1)
        # println(i)

        x=train_x[:,:,i];
        x=reshape(x,784,1);
        y_hat=forward(x, net);
        y=reshape(train_y_one_hot[:,i], 10, 1);

        epoch_train_correct+=(argmax(y_hat)==argmax(y))

        loss_grad=loss_grad.+(xe_loss_derivative(y_hat, y)./mb_size)
        loss=loss.+(xe_loss(y_hat,y)./mb_size)
        
        if mb_counter%mb_size==0 #update the net, store loss, reset counters
            gradients=full_backward_pass(loss_grad, net)
            update(gradients, 10.0, net)
            append!(epoch_train_loss, loss)

            mb_counter=0
            loss_grad=0
            loss=0
        end

        mb_counter+=1

    end
    append!(avg_epoch_train_loss,sum(epoch_train_loss)/length(epoch_train_loss))
    epoch_train_loss=[]
    append!(avg_epoch_train_acc, epoch_train_correct/size(train_y,1))
    epoch_train_correct=0.0

    for i in 1:size(test_y,1)
        x=test_x[:,:,i];
        x=reshape(x,784,1);
        y_hat=forward(x, net);
        y=reshape(test_y_one_hot[:,i], 10, 1);
        epoch_test_correct+=(argmax(y_hat)==argmax(y))
        append!(epoch_test_loss, xe_loss(y_hat,y))
    end
    append!(avg_epoch_test_loss, sum(epoch_test_loss)/length(epoch_test_loss))
    epoch_test_loss=[]
    append!(avg_epoch_test_acc, epoch_test_correct/size(test_y,1))
    epoch_test_correct=0.0

end

#%% Plot
using Plots
plot([avg_epoch_train_loss, avg_epoch_test_loss], lw=2, label=["Train" "Test"])
xlabel!("Epoch")
ylabel!("XE Loss")

#%%
plot([avg_epoch_train_acc, avg_epoch_test_acc], lw=2, label=["Train" "Test"])
xlabel!("Epoch")
ylabel!("Accuracy")