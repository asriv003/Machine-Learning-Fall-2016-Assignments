function l3_h1(test,numrep,B,X,Y)
%Abhishek Kumar Srivastava
%Student Id: 861307778
%November 8, 2016
%CS 229
%PS5
%Layers = 3, Hidden units = 1
    clc
    %weight decay
    lambda = [0.001,0.01,0.1];
    for l = 1:3
        total_loss_final = 0;
        W_1_final = zeros(1,3);
        W_2_final = zeros(1,2);
        W_3_final = zeros(1,2);
        for n = 1:numrep
            
            figure(3*(test-1)+l);

            %add bias at each layer
            A_1 = [B,X];    %80x3

            %bias + weights in range [-1,1] at 1st layer
            W_1 = [rand,(rand(1,2).*2 -1)];  %1x3

            %bias + weights at 2nd layer with 1 hidden unit
            W_2_1 = [rand,(rand(1,1).*2 -1)];    %1x2

            %bias + weights at 3rd layer with 1 hidden unit
            W_3_1 = [rand,(rand(1,1).*2 -1)];    %1x2

            %Loss Value
            total_loss_new = 0;
            %learning rate
            eta = 1;

            [H,A_3_1,A_2_1] = forward_prop_with_3_layer(A_1,W_1,W_2_1,W_3_1);
            total_loss_old = compute_loss_function_with_3_layer(Y,H,W_1,W_2_1,W_3_1,lambda(l));

            while abs(total_loss_old - total_loss_new) > 1e-8*abs(total_loss_old)
                %error at output layer
                total_loss_old = compute_loss_function_with_3_layer(Y,H,W_1,W_2_1,W_3_1,lambda(l));
                D_4 = H-Y;
                [D_3_1,D_2_1] = backward_prop_with_3_layer(D_4,A_3_1,W_3_1,A_2_1,W_2_1);

                %saving temporary values.
                tW_1 = W_1;
                tW_2_1 = W_2_1;
                tW_3_1 = W_3_1;
                tH = H;
                tA_2_1 = A_2_1;
                tA_3_1 = A_3_1;
                %gradient descent
                [W_1,W_2_1,W_3_1] = gradient_descent_with_3_layer(A_1,A_2_1,A_3_1,D_2_1,D_3_1,D_4,W_1,W_2_1,W_3_1,eta,lambda(l));
                %forward propagation with new weights
                [H,A_3_1,A_2_1] = forward_prop_with_3_layer(A_1,W_1,W_2_1,W_3_1);

                %calculate loss function with the change in weights
                total_loss_new = compute_loss_function_with_3_layer(Y,H,W_1,W_2_1,W_3_1,lambda(l));

                %loss function comparision
                if total_loss_old > total_loss_new
                    eta = eta*1.05;

                end
                if total_loss_old < total_loss_new
                    %reversing weights & reducing learning rate
                    W_1 = tW_1;
                    W_2_1 = tW_2_1;
                    W_3_1 = tW_3_1;
                    H = tH;
                    A_2_1 = tA_2_1;
                    A_3_1 = tA_3_1;
                    eta = eta*0.5;
                end
            end
            
            if(n == 1)
                W_1_final = W_1;
                W_2_final = W_2_1;
                W_3_final = W_3_1;
                total_loss_final = total_loss_new;
            else
                if(total_loss_new < total_loss_final)
                    W_1_final = W_1;
                    W_2_final = W_2_1;
                    W_3_final = W_3_1;
                    total_loss_final = total_loss_new;
                end
            end
        end
        plotclassifier(X,Y,@(X) myclassifier(X,W_1_final(:,2:3),W_2_final,W_3_final),0.5,0);
        if(lambda(l) == 0.001)
            title('layer = 3, hidden units = 1, \lambda = 0.001');
        elseif(lambda(l) == 0.01)
            title('layer = 3, hidden units = 1, \lambda = 0.01');
        elseif(lambda(l) == 0.1)
            title('layer = 3, hidden units = 1, \lambda = 0.1');
        end
    end
end

function [H, A_3, A_2] = forward_prop_with_3_layer(A_1,W_1,W_2,W_3)
    G_1 = A_1*W_1';

    %sigmoid function
    A = 1.0 ./ ( 1.0 + exp(-G_1));
    A_2 = [ones(80,1),A];

    G_2 =  A_2*W_2';
    B = 1.0 ./ ( 1.0 + exp(-G_2));
    A_3 = [ones(80,1),B];
    h = A_3*W_3';
    %sigmoid function
    H = 1.0 ./ ( 1.0 + exp(-h));
end

function [D_3, D_2] = backward_prop_with_3_layer(D_4,A_3,W_3,A_2,W_2)
    D_3 = (A_3.*(1-A_3)).*(D_4*W_3);
    D_2 = (A_2.*(1-A_2)).*(D_3(:,2)*W_2);
end

function [W_1, W_2, W_3] = gradient_descent_with_3_layer(A_1,A_2,A_3,D_2,D_3,D_4,W_1,W_2,W_3,eta,lambda)
    %New weight values
    W_1 = W_1 - eta.*(D_2(:,2)'*A_1) - lambda.*eta.*W_1;
    W_2 = W_2 - eta.*(D_3(:,2)'*A_2) - lambda.*eta.*W_2;
    W_3 = W_3 - eta.*(D_4'*A_3) - lambda.*eta.*W_3;
end

function total_loss = compute_loss_function_with_3_layer(Y,H,W_1,W_2,W_3,lambda)
    %calculate loss function
    loss = -((Y.*log(H))+((1-Y).*log(1-H)));
    total_loss = sum(loss);
    %add regularization in total loss
    R = (lambda./2).*(sum(sum(W_1.^2)) + sum(sum(W_2.^2)) + sum(sum(W_3.^2)));
    total_loss = (total_loss + R)./80;
end

function Y = myclassifier(X,W_1,W_2,W_3)
    G_1 = X*W_1';

    %sigmoid function
    A = 1.0 ./ ( 1.0 + exp(-G_1));
    A_2 = [ones(size(G_1,1),1),A];

    G_2 =  A_2*W_2';
    B = 1.0 ./ ( 1.0 + exp(-G_2));
    A_3 = [ones(size(G_2,1),1),B];
    h = A_3*W_3';
    %sigmoid function
    Y = 1.0 ./ ( 1.0 + exp(-h));
end
