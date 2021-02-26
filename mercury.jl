#%%
using LinearAlgebra
using MLDatasets
using Plots

include("utils.jl");
include("components.jl");
include("training.jl");


#%% -- Instantiate the network -- %%#
# The net is simply a dictionary containing parameters,
# activations pre- (A) and post- (Z) activation function.
net=Dict("Layers"=>[], "A"=>[], "Z"=>[])

# [[layer1_inputdim, layer1_outputdim] [layer2_inputdim, layer2_outputdim] [layer3_inputdim, layer3_outputdim] ... ]
dims=[[28^2, 32] [32, 32] [32, 10]]
layers=[]
for i in 1:size(dims,2)-1
    append!(layers, [Dense(dims[1,i], dims[2,i], ReLU())])
end

head=[Dense(dims[1, size(dims,2)], dims[2, size(dims,2)], Softmax())]
append!(layers, head);
net["Layers"]=layers;


#%% -- Initialiset the training data -- %%#
train_x, train_y = MNIST.traindata(Float64);
test_x, test_y = MNIST.testdata(Float64);

train_y_one_hot=zi_one_hot_encode(train_y);
test_y_one_hot=zi_one_hot_encode(test_y);


#%% -- Train -- %%#
mb_size=32
epochs=50
lr=0.01

results=train(net, mb_size, lr, epochs, train_x, test_x, train_y_one_hot, test_y_one_hot);


#%% -- Plot -- %%#
plot([results["training_loss"], results["test_loss"]], lw=2, label=["Train" "Test"], legend=:topright)
xlabel!("Epoch")
ylabel!("Loss")

#%%
plot([results["training_acc"], results["test_acc"]], lw=2, label=["Train" "Test"], legend=:bottomright)
xlabel!("Epoch")
ylabel!("Accuracy")