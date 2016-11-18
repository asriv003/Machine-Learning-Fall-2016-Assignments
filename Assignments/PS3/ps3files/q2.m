function q2()
%Abhishek Kumar Srivastava
%Student Id: 861307778
%October 13, 2016
%CS 229
%PS2
%Testing of Q2
clc;
hold on;
X = load('simple1X.data','-ascii');
Y = load('simple1Y.data','-ascii');
%plot(X(:,1),X(:,2),'or');
%plot(X(11:20,1),X(11:20),'ob');
C = .5;
[w b] = learnsvm(X,Y,C);
%drawline(w, b);
disp(w);
disp(b);
hold off;
end

%for simple1X.data & simpleY.data 
%c = .1
%   -0.2024 // -1.0139
%   -0.6804 // -2 0320

%    0.2517 // 0

%c = 1
%   -0.2239 // -110.3863
%   -0.9211 // -212.1957
%
%    0.2369 // 0

%c = 10
%   -0.2239 //-158.6627
%   -0.9211 //-239.2739

%    0.2369 //0

%for simple2X.data & simple2Y.data
%c = .1
%    0.4951 //2.0884
%    0.5310 //1.6538

%   -0.2636 //0
%c = 1
%    1.0981 //199.8407
%    1.3570 //156.3850

%   -1.4047 //0

%c = 10
%    2.3315 //261.2759
%    2.9964 //157.9087

%   -3.6183 // 0

