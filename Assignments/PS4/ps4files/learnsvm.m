function [alpha,b] = learnsvm(X,Y,C,kernel)
% [alpha,b] = learnsvm(X,Y,C,kernel)
% X is an m-by-n matrix of m data points (each of n dimensions)
% Y is an m-by-1 vector of labels (+1 or -1)
% C is a positive scalar.  Larger C means smaller margins, but more
%   points on the correct side of the margin
% kernel is a function of two arguments
%      if A is r-by-n and B is s-by-n
%      kernel(A,B) is r-by-s in which the i,j element is
%        the kernel applied to A(i,:) and B(j,:)
%      Thus, the simple "linear kernel" is the function
%           @(A,B) A*B'
%      and a polynomial kernel of degree d (and constant c) is
%           @(A,B) (A*B' + c).^d

	tol = 1e-3;
	K = kernel(X,X);
	n = size(Y,1);
	G = K.*(Y*Y');
	alpha = zeros(n,1);
	b = 0;
	c=1;
	nch = 0;
	exall = 1;
	while nch>0 | exall
		nch = 0;
		if (exall)
			for i=1:length(alpha)
				nch = nch + picksecond(i);
			end
		else
			for i=1:length(alpha)
				if alpha(i)<C && alpha(i)>0
					nch = nch + picksecond(i);
				end
			end
		end
		if exall
			exall=0;
		elseif nch==0
			exall=1;
		end
	end

	function nch = picksecond(i)
		E = ((K*(alpha.*Y)) + b) - Y;
		l = E.*Y;
		nch = 0;
		if (l(i) < -tol && alpha(i)<C) || (l(i) > tol && alpha(i)>0)
			if sum(alpha>0 & alpha<C) > 1
				%disp('a');
				[tmp,j] = max(abs(E-E(i)));
				nch = update(i,j);
			end
			if ~nch
				%disp('b');
				jdel = randi(length(alpha),1);
				for jj=1:length(alpha)
					j = mod(jj+jdel,length(alpha))+1;
					if j~=i && alpha(j)<C && alpha(j)>0
						if update(i,j)
							nch = 1;
							break;
						end
					end
				end
			end
			if ~nch
				%disp('c');
				jdel = randi(length(alpha),1);
				for jj=1:length(alpha)
					j = mod(jj+jdel,length(alpha))+1;
					if j~=i
						if update(i,j)
							nch = 1;
							break;
						end
					end
				end
			end
		end
	end

	function nch = update(i,j)
		% write this function!
		% note that you have access to the variables from above
		% including K, G, Y, and alpha
        E = ((K*(alpha.*Y)) + b) - Y;
		% this function needs to return whether it made a change
		% to do this, check to see that the change in alpha(i)
		% was greater than 1e-10 times (the sum of the old and 
		% new values plus 1e-10)
		% if it was, return 1 (you made a change!)
		% if it was not, return 0 (no real change)
		%
        nch = 0;
        a1 = alpha(i) - ((Y(i).*(E(j)-E(i)))./(2.*K(i,j)-K(i,i)-K(j,j)));
		if abs(a1-alpha(i)) >= (1e-10*(a1+alpha(i)) + 1e-10)
            a2 = alpha(j) + (alpha(i) - a1)*(Y(i)*Y(j));
            nch = 1;
        end
        % if there is a change, update b by the following statement
		if nch
            b = findb(K*(alpha.*Y),Y);
            alpha(i) = a1;
            alpha(j) = a2;
        end
	end

end
function b = findb(f,Y)

	f(Y>0) = f(Y>0)-1;
	f(Y<0) = f(Y<0)+1;
	[f,i] = sort(f);
	Y = Y(i);
	diffY = (1:(length(Y)+1))'-ceil((length(Y)+1)/2);
	difff = [f;0] - [0;f];
	diff = diffY.*difff;
	v = cumsum(diff);
	v = v(2:end-1);
	[bestv,ii] = min(v);
	if v(ii+1)==bestv
		b = -(f(ii+1)+f(ii+2))/2;
	else
		b = -f(ii+1);
	end
end
