function q2()
%Abhishek Kumar Srivastava
%Student Id: 861307778
%October 13, 2016
%CS 229
%PS2
%Solution of Q2

    %clear up workspaces
    clc;
    hold on;
    lambda = [0.001 0.1 10];    %lambda variations
    %for each lambda value
    for l = 1:3
        %for storing ridge values
        b = zeros(6,1000); %6x1000
        for i = 1:1000
            %training data
            x = sort(unifrnd(-1,1,1,10));   %1x10
            %features powers of x from 0 through 5
            x_pow = [x.^0;x.^1;x.^2;x.^3;x.^4;x.^5]; %6x10

            %true function
            y = tan((pi*x)/3)+(x-0.5).^2;   %1x10
            %for testing purpose
            %y = sin(pi*x);

            %added gaussian noise with standard deviation of 0.5
            y_n = y + 0.5*randn;    %1x10
 
            %ridge function
            %y_n' & x_pow' since no of rows should be same for X & Y
            %b(:,i) = ridge(y_n',x_pow',lambda(l)); %6x1000
            b(:,i) = pinv(x_pow*x_pow' + lambda(l).*eye(6))*x_pow*y_n';
        end

        %subplots
        subplot(1,3,l);
        hold on;
        %testing data
        x_t = sort(unifrnd(-1,1,1,200)); %1x100
        %feature powers of x_t from 0 through 5
        x_pow_test = [x_t.^0;x_t.^1;x_t.^2;x_t.^3;x_t.^4;x_t.^5]; %6x100
        
        %predicted values of y from ridge regression
        y_pow = b'*x_pow_test;  % 1000x6 * 6x100 = 1000x100
        
        %100 random testing plot from y_pow 
        r = randi(1,900);
        y_t = y_pow(r:r+100,:);
        plot(x_t,y_t','r');

        %average plot
        h1 = plot(x_t,mean(y_pow),'b');
        
        %true plot
        y_true = tan((pi*x_t)/3)+(x_t-0.5).^2;
        %y_true = sin(pi*x_t);
        h2 = plot(x_t,y_true,'k');
        
        %subplot attributes
        xlabel('x');
        ylabel('y');
        set(h1,'linewidth',1.5);
        set(h2,'linewidth',1.5);
        axis([-1 1 -0.5 4.5]);
        if(l==1)
            title('λ = 0.001');
        elseif(l==2)
            title('λ = 0.1');
        elseif(l==3)
            title('λ = 10');
        end
    end
end
