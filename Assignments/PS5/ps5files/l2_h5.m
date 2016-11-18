function l2_h5(test,numrep,B,X,Y)
%Abhishek Kumar Srivastava
%Student Id: 861307778
%November 8, 2016
%CS 229
%PS5
%Layer = 2, Hidden units = 5
    clc
    %weight decay
    lambda = [0.001, 0.01, 0.1];
    %layer = 2, hidden units = 5
    for l = 1:3
        total_loss_final = 0;
        W_1_final = zeros(5,3);
        W_2_final = zeros(1,6);
        for n = 1:numrep
            
            figure(3*(test-1)+l);

            %add bias at each layer
            A_1 = [B,X];    %80x3

            %bias + weights in range [-1,1] at 1st layer
            W_1 = [ones(5,1),(rand(5,2).*2 -1)];  %5x3

            %bias + weights at 2nd layer with 1 hidden unit
            W_2_5 = [1,(rand(1,5).*2 -1)];    %1x6

            %Loss Value
            total_loss_new = 0;
            %learning rate
            eta = 1;

            [H, A_2_5] = forward_prop_with_2_layer(A_1,W_1,W_2_5);

            total_loss_old = compute_loss_function_with_2_layer(Y,H,W_1,W_2_5,lambda(l));

            while abs(total_loss_old - total_loss_new) > 1e-8*abs(total_loss_old)
                %error at output layer
                total_loss_old = compute_loss_function_with_2_layer(Y,H,W_1,W_2_5,lambda(l));
                D_3 = H-Y;
                D_2_5 = backward_prop_with_2_layer(D_3,A_2_5,W_2_5);

                %saving temporary values.
                tW_1 = W_1;
                tW_2_5 = W_2_5;
                tH = H;
                tA_2_5 = A_2_5;
                %gradient descent
                [W_1,W_2_5] = gradient_descent_with_2_layer(A_1,A_2_5,D_2_5,D_3,W_1,W_2_5,eta,lambda(l));
                %forward propagation with new weights
                [H, A_2_5] = forward_prop_with_2_layer(A_1,W_1,W_2_5);

                %calculate loss function with the change in weights
                total_loss_new = compute_loss_function_with_2_layer(Y,H,W_1,W_2_5,lambda(l));

                %loss function comparision
                if total_loss_old > total_loss_new
                    eta = eta*1.05;

                end
                if total_loss_old < total_loss_new
                    %reversing weights & reducing learning rate
                    W_1 = tW_1;
                    W_2_5 = tW_2_5;
                    H = tH;
                    A_2_5 = tA_2_5;
                    eta = eta*0.5;
                end
            end
            
            if(n == 1)
                W_1_final = W_1;
                W_2_final = W_2_5;
                total_loss_final = total_loss_new;
            else
                if(total_loss_new < total_loss_final)
                    W_1_final = W_1;
                    W_2_final = W_2_5;
                    total_loss_final = total_loss_new;
                end
            end
        end
        plotclassifier(X,Y,@(X) myclassifier(X,W_1_final(:,2:3),W_2_final),0.5,0);
        if(lambda(l) == 0.001)
            title('layer = 2, hidden units = 5, \lambda = 0.001');
        elseif(lambda(l) == 0.01)
            title('layer = 2, hidden units = 5, \lambda = 0.01');
        elseif(lambda(l) == 0.1)
            title('layer = 2, hidden units = 5, \lambda = 0.1');
        end
    end
end

function [H, A_2] = forward_prop_with_2_layer(A_1,W_1,W_2)
    G = A_1*W_1';

    A = zeros(size(G));
    %sigmoid function
    A = 1.0 ./ ( 1.0 + exp(-G));
    A_2 = [ones(80,1),A];

    h = A_2*W_2';
    %sigmoid function
    H = 1.0 ./ ( 1.0 + exp(-h));
end

function D_2 = backward_prop_with_2_layer(D_3,A_2,W_2)
    D_2 = (A_2.*(1-A_2)).*(D_3*W_2);
end

function [W_1, W_2] = gradient_descent_with_2_layer(A_1,A_2,D_2,D_3,W_1,W_2,eta,lambda)
    %New weight values
    W_1 = W_1 - eta.*(D_2(:,2)'*A_1);
    W_2 = W_2 - eta.*(D_3'*A_2);
end

function total_loss = compute_loss_function_with_2_layer(Y,H,W_1,W_2,lambda)
    %calculate loss function
    loss = -((Y.*log(H))+((1-Y).*log(1-H)));
    total_loss = sum(loss);
    %add regularization in total loss
    R = (lambda./2).*(sum(sum(W_1.^2)) + sum(sum(W_2.^2)));
    total_loss = (total_loss + R)./80;
end

function Y = myclassifier(X,W_1,W_2)
    G = X*W_1';

    A = zeros(size(G));
    %sigmoid function
    A = 1.0 ./ ( 1.0 + exp(-G));
    A_2 = [ones(size(G,1),1),A];

    h = A_2*W_2';
    %sigmoid function
    Y = 1.0 ./ ( 1.0 + exp(-h));
end
