% Simple script to illustrate features of matlab
% try entering the commands by hand and observing the results

% how to get help: (you can easily pick up all of Matlab with this)
help % list of help topics
help eig % help on the command eig
lookfor diagonal % list of all commands whose help mentions "diagonal"
doc eig % graphical help application, started on topic "eig"

% all (well, not really) variables are matrices:
A = [1 2; 3 4]
B = [5 6; 7 8]

% placing a semicolon after a command supresses the output:
C = [1 2 3; 4 5 6];
C

% standard +, -, and * operators for matrices exist:
A+B
A-B
A*B % note that this is matrix multiplication

% vectors are n-by-1 or 1-by-n matrices:
x = [10;11]
A*x

% scalars are 1-by-1 matrices:
s = 0.5
A*s
s*A

% other operators
A' %(' = transpose)
A.*B % .* is element-wise multiplication
A./B % ./ is element-wise division

% the colon operator:
4:10 % same as row vector [4 5 6 7 8 9 10]
4:2:10 % same as row vector [4 6 8 10]

% selection: (first element is 1, NOT 0)
% five methods: (well, some are subsets of others)
% 1 - simple scalars -- like a multi-dimensional array in any other language:
A(1,2)
% 2 - using a single : to specify all of one (or more) dimensions:
A(:,2) % gets the entire 2nd column
% 3 - using a vector of indices to select those elements of a dimension:
C([1 2],[1 3]) % gets elements C(1,1) C(1,2) C(2,1) C(2,3)
C(1:2,1:2) % gets elements C(1,1) C(1,2) C(2,1) and C(2,2)
		% in their relative positions
C(2:end,:) % "end" is special in this context and refers to the last
	% index along a particular dimension
% 4 - using a vector of booleans (must be same lenght as corresponding
%     dimension
C([true true],[false true false])
% 5 - using a boolean matrix the same size as the matrix
A([true true; false true])
	% note that this one does select a submatrix, but rather puts
	% all of the relevant elements into a vector

% this selection can be written to:
D = A
D(D<2 | D>3) = -10

% other useful functions
eig(A) % eigenvalues (or vectors) of matrix A
det(A) % determinant of A
repmat(A,[2 1]) % duplication A twice along first dimension
repmat(A,[3 2])
repmat(A,[1 1 3) % matrices can have more than 2 dimensions
permute(C,[2 1]) % same as transpose -- can do other dimension reorderings
zeros(4,5) % constructs matrix of zeros
ones(4,5) % same for ones
size(C) % returns vector of sizes of C
size(C,2) % returns size of C's 2-nd dimension
rand(3,4) % 3-by-4 matrix of (independent) random values, drawn uniformly
		% between 0 and 1
randn(2,3) % 2-by-3 matrix of (independent) random values, drawn from
		% a normal (Gaussian) distribution with mean 0 and variance 1

% control flow:
% standard for, while, if statments exist:
% sums up the numbers from 1 to 4
% standard notation for for is for <var> = <vector>
% and variable takes on each of the values in vector, one by one
% try to avoid for loops for speed (although it isn't terrible)
for i=1:4,
	b = b+i;
end;

% if statement
% statement below incriments b is a is positive (otherwise, it decrements)
if a>0
	b = b+1;
else
	b = b-1;
end;
% chained if statements use "elseif" as a keyword:
if a>0
	b = b+1;
elseif a<0
	b = b-1;
else
	b = 0;
end;

% pausing and debugging:
% the command "pause" pauses for user input (just a "return")
% the command "pause(s)$ pauses for s seconds (fractional s is okay
%   as is s=0).  This also allows for any figures to be updated
% the command keyboard dumps Matlab into debugging mode
% (this can also be done by selecting the "stop sign" next to the
% line if you are editing with Matlab's editor)

% functions are declared in files with the same name.  So, the
% text below between %----- should be put in the file "myfn.m"
% note that comments immediately following the first "function" line
% are interpreted to be the "help" information for the function
% arguments are passed by value (not reference), so changing the formal
% parameter "X" below % would not change the actual parameter

%-----
function [cpos,cneg] = myfn(X)
% [cpos,cneg] = myfn(X) returns two counts: the number of
% postive elements in X (cpos) and the number of negative elements in X
% (cneg)
cpos = sum(sum(X>0));
cneg = sum(sum(X<0));
% or, the long way...
%cpos = 0;
%cneg = 0;
%[n,m] = size(X);
%for i=1:n
%	for j=1:m
%		if (X(i,j)<0)
%			cneg = cneg+1;
%		elseif (X(i,j)>0)
%			cpos = cpos+1;
%		end;
%	end;
%end;
%-----

% subfunctions (that are only visible to the main function of the file)
% may be declared after the main function (whose name matches the file's)

%plotting:

plot([2 3 4],[10 12 9]); % plots the second vector against the first
plot([2 3 4],[10 12 9],'k:'); % same, but use blacK dotted line
% see "help plot" for more options
h = plot([2 3 4],[10 12 9],'b-'); % returns a "handle"
get(h) % lists all properties
set(h,'LineWidth',3); % make it a thick line

figure; % create a new figure
figure(1); % make the old figure the current figure
figure(2); % switch back to the new figure

x = sort(rand(10,1)*2*pi) % create a random set of points on [0,2PI]
y = sin(x) + randn(10,1)*0.1 % y = sin(x) + noise
h1 = plot(x,y,'bo'); % plot with blue circles
hold on; % prevent future "plot" commands from erasing current figure

set(h2,'LineWidth',3);
axis([0 2*pi -1 1]); % override default axis limits
legend('training data','true f'); % give labels to each "plot" command
hold off;  % now future "plot" commands will erase the plot


% look at other commands:
% line, subplot, plot3, surf, contour, quiver, image, imagesc


% saving & loading:
save % saves current workspace to matlab.mat
load % loads current worksapce from matlab.mat
help load % gives other options
help save % gives other options

% see path and setpath to control where matlab looks for commands
