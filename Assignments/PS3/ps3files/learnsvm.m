function [w,b] = learnsvm(X,Y,C)
%Abhishek Kumar Srivastava
%Student Id: 861307778
%October 21, 2016
%CS 229
%PS3
%Solution of Q2
    %size of X
    [m n] = size(X);
    %weight vector
    w = zeros(n,1);
    b = 0;
    ss = 1/(C*m);

    %while the step size is greater than 10^−6 /C*m
    while ss > 1.0e-6/(C*m)
        
        %calculate the total loss
        L = 0.5*(w'*w); %Base Value
        N = 0;      %Noise 

        for i = 1:m
            if(Y(i,1)*(w'*X(i,:)'+ b) < 1)  %i.e yi.f(xi) < 1
                N = N + Y(i,1)*(w'*X(i,:)'+ b); %collect the summation
            end
        end
        
        %total prev loss function
        L = L + N;

        for i = 1:100
            %update for yi.f(xi) < 1 for each point datapoint
            for j = 1:m
                tw = w;
                tb = b;
                if(Y(j,1)*(w'*X(j,:)'+ b) < 1)  %i.e yi.f(xi) < 1
                    tw = tw - ss.*(tw - C.*(X(j,:)'*Y(j,1)));
                    tb = tb + ss.*(C*Y(j,1));
                end
                w = tw;
                b = tb;
                %else wont update anything
            end

        end
        
        %New Loss function
        new_L = 0.5*(w'*w);
        new_N = 0; 
        
        for i = 1:m
            if(Y(i,1)*(w'*X(i,:)'+ b) < 1)  %i.e yi.f(xi) < 1
                new_N = new_N + Y(i,1)*(w'*X(i,:)'+ b); %collect the summation
            end
        end
        new_L = new_L + new_N;

        if(new_L < L)
            ss = ss + 0.05*ss;
        else
            ss = ss - 0.5*ss;
        end
    end
end