function ps7()
%Abhishek Kumar Srivastava
%Student Id: 861307778
%November 22, 2016
%CS 229
%PS7
    clc;
    I = load('spamtrain.ascii','-ascii');
    X = I(:,1:57);   %:x57
    Y = I(:,58);     %:x1
    T = load('spamtest.ascii','-ascii');
    tX = T(:,1:57);   %:x57
    tY = T(:,58);     %:x1
    trees = floor(logspace(0,2+log10(5),10));
    max_tree = trees(10);
    tr_bag = cell(max_tree,3);
    tr_boost = cell(max_tree,3);
    wt_boost = zeros(max_tree,1);
    err_boost = zeros(max_tree,1);
    test_error_bag = zeros(1,size(trees,2));
    test_error_boost = zeros(1,size(trees,2));
    test_error_bag_reweighted = zeros(1,size(trees,2));
    test_error_boost_reweighted = zeros(1,size(trees,2));
    alpha  = ones(size(Y,1),1);
    lambda = logspace(-4,-1,10);
    %to store the value of trees
    val_bag = zeros(max_tree,size(tY,1));
    val_boost = zeros(max_tree,size(tY,1));
    trees_bag = zeros(10,1);
    trees_boost = zeros(10,1);
    d = 2;
    for i = 1:10
        %sprintf('depth: %d trees: %d\n',d,trees(i));
        remain = 1;
        if i >1
            remain = trees(i) - trees(i-1);
        end
        for sz = 1:remain
            %bootstrap sampling for bagging
            S = datasample(I,size(Y,1));
            B_X = S(:,1:57);   %80x57
            B_Y = S(:,58);     %80x1
            %decision tree for bagging and boosting
            t_bag = traindt(B_X,B_Y,d);
            t_boost = traindtw(X,Y,alpha,d);
            %storing and modifying alpha and weights for boosting
            if i > 1
                tr_bag(trees(i-1) + sz,:) = t_bag;
                tr_boost(trees(i-1) + sz,:) = t_boost;
                err_boost(trees(i-1) + sz) = error_calculation(X,Y,t_boost,1,alpha);
                wt_boost(trees(i-1) + sz) = log(1-err_boost(trees(i-1) + sz)) - log(err_boost(trees(i-1) + sz));
                alpha = update_alpha(X,Y,t_boost,alpha,wt_boost(trees(i-1) + sz));
            else
                tr_bag(sz,:) = t_bag;
                tr_boost(sz,:) = t_boost;
                err_boost(sz) = error_calculation(X,Y,t_boost,1,alpha);
                wt_boost(sz) = log(1-err_boost(sz)) - log(err_boost(sz));
                alpha = update_alpha(X,Y,t_boost,alpha,wt_boost(sz));
            end
        end

        test_error_bag(i) = classification_error_bagging(tX,tY,tr_bag(1:trees(i),:),trees(i));
        test_error_boost(i) = classification_error_boosting(tX,tY,tr_boost(1:trees(i),:),trees(i),wt_boost(1:trees(i),:));
    end
    %bagging & boosting reweighted
    for i = 1:max_tree
        val_bag(i,:) = (dt(tX,tr_bag(i,:)))';
        val_boost(i,:) = (dt(tX,tr_boost(i,:)))';
    end
    
    for l = 1:10

        [w_bag,w0_bag] = lassoglm(val_bag', tY == 1, 'binomial', 'Standardize', 0, 'Lambda', lambda(l) ) ;
        [w_boost,w0_boost] = lassoglm(val_boost', tY == 1, 'binomial', 'Standardize', 0, 'Lambda', lambda(l) ) ;
        w_bag = 2*w_bag ;
        w0_bag = 2*w0_bag.Intercept -1;
        w_boost = 2*w_boost ;
        w0_boost = 2*w0_boost.Intercept -1;
        num_bag = 0;
        num_boost = 0;
        for n = 1:size(w_bag,1)
            if w_bag(n) ~= 0
               num_bag = num_bag + 1;
            end
            if w_boost(n) ~= 0
               num_boost = num_boost + 1;
            end
        end
        trees_bag(l) = num_bag;
        trees_boost(l) = num_boost;
        test_error_bag_reweighted(l) = classification_error_reweighted(val_bag,tY,w_bag,w0_bag);
        test_error_boost_reweighted(l) = classification_error_reweighted(val_boost,tY,w_boost,w0_boost);
    end
    figure(1);
    semilogx(trees,test_error_bag);
    hold on;
    semilogx(trees,test_error_boost);
    %number of trees calculate
    semilogx(trees_bag,test_error_bag_reweighted);
    semilogx(trees_boost,test_error_boost_reweighted);
    
    title('Error Rate Comparision');
    xlabel('number of trees');
    ylabel('error rate');
    legend('bagging','boosting','bagging reweighted','boosting reweighted');
end

function err = classification_error_bagging(X,Y,t,B)
    tY  = zeros(size(Y,1),1);
    for i = 1:B
        tY = tY + dt(X,t(i,:));
    end
    
    tY = tY./B;
    [m n] = size(tY);
    for j = 1:m
        if tY(j) > 0
            tY(j) = 1;
        else
            tY(j) = -1;
        end
    end
    err = 0;
    for j = 1:m
        if tY(j) ~= Y(j)
            err = err + 1;
        end
    end
    err = err./size(Y,1);
end


function err = classification_error_boosting(X,Y,t,B,w)
    tY  = zeros(size(Y,1),1);
    for i = 1:B
        tY = tY + dt(X,t(i,:)).*w(i);
    end
    
    [m n] = size(tY);
    for j = 1:m
        if tY(j) > 0
            tY(j) = 1;
        else
            tY(j) = -1;
        end
    end
    err = 0;
    for j = 1:m
        if tY(j) ~= Y(j)
            err = err + 1;
        end
    end
    err = err./size(Y,1);
end

function err = classification_error_reweighted(val_tree,Y,w,w0)
    tY = val_tree'*w + w0;
    [m n] = size(Y);
    err = 0;
    for j = 1:m
        if tY(j) > 0
            tY(j) = 1;
        else
            tY(j) = -1;
        end
        if tY(j) ~= Y(j)
            err = err + 1;
        end
    end
    err = err./m;
end

function err = error_calculation(X,Y,t,B,alpha)
    tY  = zeros(size(Y,1),1);
    for i = 1:B
        tY = tY + dt(X,t(i,:));
    end
    
    [m n] = size(tY);
    for j = 1:m
        if tY(j) > 0
            tY(j) = 1;
        else
            tY(j) = -1;
        end
    end
    al = 0;
    for j = 1:m
        if tY(j) ~= Y(j)
            al = al + alpha(j);
        end
    end
    err = al./sum(alpha);
end

function al = update_alpha(X,Y,t,alpha,w)
    %testY  = zeros(size(Y,1),1);
    testY = dt(X,t);
    [m n] = size(testY);
    al = alpha;
    for j = 1:m
        if testY(j) ~= Y(j)
            al(j) = alpha(j).*exp(w);
        end
    end
end