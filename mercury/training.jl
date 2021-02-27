include("backpropagation.jl")

# NOTE:
# unlike the blog post, here we use y_hat as predictions and y as true values.

# This function defines the training loop for a neural network.
function train(net, mb_size, lr, epochs, train_x, test_x, train_y_one_hot, test_y_one_hot)

    avg_epoch_train_loss=[]
    avg_epoch_test_loss=[]
    avg_epoch_train_acc=[]
    avg_epoch_test_acc=[]

    for epoch in 1:epochs

        epoch_train_loss=[]
        epoch_test_loss=[]
        epoch_train_correct=0.0
        epoch_test_correct=0.0

        mb_counter=1;
        loss=0.0;

        d=-1
        
        println("Running epoch ", epoch)

        for i in 1:size(train_y,1)

            # Grab training data point
            x_tr = train_x[:, :, i];
            x_tr = reshape(x_tr, 784, 1);
            y_hat_tr = forward(x_tr, net);
            y_tr = reshape(train_y_one_hot[:, i], 10, 1);

            # track loss and loss gradient
            loss_grad = xe_loss_derivative(y_hat_tr, y_tr)./mb_size

            # accumulate the parameter gradient
            d = accumulate_gradient(loss_grad, net, d)

            # accumulate loss and accuracy
            loss = loss.+(xe_loss(y_hat_tr, y_tr)./mb_size)
            epoch_train_correct+=(argmax(y_hat_tr) == argmax(y_tr))
            
            # have we reached the end of a minibatch?
            if mb_counter%mb_size==0 #update the net, store loss, reset counters

                # perform a weight update on average gradient / loss
                update(d, lr, net)
                
                # average over minibatches so we don't have too massive lists
                append!(epoch_train_loss, loss)

                # reset gradient, counter, and loss accumulator
                d=-1;
                mb_counter=0;
                loss=0.0;

            end

            mb_counter+=1

        end

        # get averages for training statistics 
        append!(avg_epoch_train_loss, sum(epoch_train_loss)/length(epoch_train_loss))
        append!(avg_epoch_train_acc, epoch_train_correct/size(train_y,1))

        for i in 1:size(test_y,1)

            # Grab test data point
            x_te=test_x[:,:,i];
            x_te=reshape(x_te,784,1);
            y_hat_te=forward(x_te, net);
            y_te=reshape(test_y_one_hot[:,i], 10, 1);

            # accumulate loss and accuracy
            append!(epoch_test_loss, xe_loss(y_hat_te,y_te))
            epoch_test_correct+=(argmax(y_hat_te)==argmax(y_te))

        end
        
        append!(avg_epoch_test_loss, sum(epoch_test_loss)/length(epoch_test_loss))
        append!(avg_epoch_test_acc, epoch_test_correct/size(test_y,1))

    end

    results=Dict("training_loss"=>avg_epoch_train_loss, 
                 "test_loss"=>avg_epoch_test_loss, 
                 "training_acc"=>avg_epoch_train_acc,
                 "test_acc"=>avg_epoch_test_acc)

    return results
    
end