function explot(trainX,trainY)
% This function picks a random weight vector and intercept (?!)
% and plots the dataset and resulting classifier.
% It isn't useful, but demonstrates how to use plotclassifier


w = randn(2,1);
w = w/sqrt(w'*w);
b = rand(1,1)*2-1;
plotclassifier(trainX,trainY,@(X) myclassifier(X,w,b));


function Y = myclassifier(X,w,b)
	Y = X*w + b;
