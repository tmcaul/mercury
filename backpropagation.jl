# Here we contain neural network forward and backward methods.
# The neural network is stored as a dictionary.

## -- Forward pass -- ##

function forward(x::Array{Float64,2}, net::Dict)::Array{Float64,2}
    # Feed the input through a network.
    # Keep track of the activations both pre- and post- activation function.
    # Store these values in the network's dictionary.
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


## -- Calculate and accumulate gradients -- ##

function calculate_gradient(dA, W, B, Z, A_prev, act_fn)
    #m = size(A_prev, 1)
    dZ = dA.*gradient(Z, act_fn)
    dW = (dZ * A_prev')# ./ m
    dB = dZ# ./ m
    dA_prev = W'*dZ
    out=[dA_prev, dW, dB]
    return out
end

function accumulate_gradient(grad, net, d)
    #m = size(grad, 1)
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

        out = calculate_gradient(dA, W, B, Z, A_prev, act_fn)     
        
        dA_prev = out[1]
        append!(dW, [out[2]])
        append!(dB, [out[3]])   
    end

    dW=reverse(dW)
    dB=reverse(dB)

    # if d doesn't yet exist, return it
    # else add to d
    if d == -1
        d=[dW, dB]
    else
        for n in 1:depth
            d[1][n]=d[1][n].+dW[n]
            d[2][n]=d[2][n].+dB[n]
        end
    end

    return d

end;

function update(gradients, learning_rate, net)
    for i in 1:length(net["Layers"])
        W = net["Layers"][i].weight-=(learning_rate)*gradients[1][i];
        B = net["Layers"][i].bias-=(learning_rate)*gradients[2][i];
    end
end;