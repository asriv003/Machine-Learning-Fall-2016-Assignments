function partc()
%Abhishek Kumar Srivastava
%Student Id: 861307778
%October 07, 2016
%CS 229
%PS1
%Solution of partC
    hold on;
    for d = [2 5 10]       % dimensions 2,5 and 10 
        sigma = eye(d)*0.2 + ones(d)*0.8;       %convergence matrix
        for m = 2:1000       % range of m 2-1000
            amd = 0;
            for i = 1:100       %average over 100 iteration
                A = randn(m,d) * sigma;     % generating random matrix of mXd
                vd = eucdist(A);       % calculate the eucledian vector distances
                vd  = vd + 10.*eye(size(vd));   %maximize the daigonal elements
                amd = amd + mean(min(vd'));     % average mean distance
            end
            amd = amd/100;      %average of mean vlaues over 100 iterations
            if(d == 2)
                plot(m,amd,'ko');     % black dots when d = 2  
            elseif(d == 5)
                plot(m,amd,'go');     %green dots when d = 5
            else
                plot(m,amd,'ro');       %red dots when d = 10
            end
        end
    end
    xlabel('m');
    ylabel('average mean distance');
    title('partc');
hold off
legend('d=2(black)','d=5(green)','d=10(red)');

function ed = eucdist(M)
[r,c] = size(M);
M1 = repmat(permute(M,[1 3 2]),[1 r 1]);
M2 = repmat(permute(M,[3 1 2]),[c 1 1]);
dist = M1-M2;
ed = sqrt(sum(dist.*dist,3)); %square root of the distances