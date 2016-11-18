function test()

    clc
    function Y = myclassifier(X,w,b)
            Y = X*w + b;
    end
    
    D = load('example1.data','-ascii');
    X = D(:,1:2);   %40x2
    Y = D(:,3);     %40x1
    
    figure(01)
    hold on
    linear_kernel = @(A,B) A*B';
    poly_kernel_1 = @(A,B) (A*B'+1).^2;
    poly_kernel_2 = @(A,B) (A*B'+1).^5;
    gauss_kernel_1 = @(A,B) exp(-(A-B).^2/(2*(1^2)));
    gauss_kernel_2 = @(A,B) exp(-(A-B).^2/(2*(5^2)));
    gauss_kernel_3 = @(A,B) exp(-(A-B).^2/(2*(10^2)));
    
    %C = 0.1
    [alpha,b] = learnsvm(X,Y,0.1,poly_kernel_2);
    w = X'*(alpha.*Y);
    subplot(1,3,1);
    %plotclassifier
    plotclassifier(X,Y,@(X) myclassifier(X,w,b));
    title('C = 0.1');
    hold off
    %C = 1
    [alpha,b] = learnsvm(X,Y,1,poly_kernel_2);
    w = X'*(alpha.*Y);
    subplot(1,3,2);
    
    %plotclassifier
    plotclassifier(X,Y,@(X) myclassifier(X,w,b));
    title('C = 1');
    %C = 10
    [alpha,b] = learnsvm(X,Y,10,poly_kernel_2);
    w = X'*(alpha.*Y);
    subplot(1,3,3);
    
    %plotclassifier
    plotclassifier(X,Y,@(X) myclassifier(X,w,b));
    title('C = 10');

end