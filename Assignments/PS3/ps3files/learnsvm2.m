function [w,b] = learnsvm2(X,Y,C)

[a,b] = solvesvm(X*X',Y,C);
w = X'*a;
