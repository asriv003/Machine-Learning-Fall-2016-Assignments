function q3()
%Abhishek Kumar Srivastava
%Student Id: 861307778
%October 23, 2016
%CS 229
%PS3
%Solution of Q3

    clc;
    %hold on;
    X = load('spamtrainX.data','-ascii');
    Y = load('spamtrainY.data','-ascii');
    
    C = logspace(-3,2,5);
    
    C_err = zeros(1,5);
    F_err = zeros(1,5);
    
    for c = 1:5
        %3-fold CV

        %1st Fold
        CV_Train_Set_1 = X(1:2000, :); %2000x57
        CV_Train_Result_1 = Y(1:2000, :); %2000x1
        
        CV_Test_Set_1 = X(2001:3000, :);   %1000x57
        CV_Test_Result_1 = Y(2001:3000, :); %1000x1
        
        
        [w1 b1] = learnsvm2(CV_Train_Set_1,CV_Train_Result_1,C(c)); %57x1, 1x1
        
        R1 =  (CV_Test_Set_1*w1 + b1); %1000x1
        
        error_count_1 = 0;
        for i = 1:1000
           if R1(i,1) >= 1 && CV_Test_Result_1(i,1) == -1
                error_count_1 = error_count_1 +1;
           elseif R1(i,1)< 1 && CV_Test_Result_1(i,1) == 1
               error_count_1 = error_count_1 +1;
           end
        end
        
        
        %2nd Fold
        CV_Train_Set_2 = [X(1:1000, :); X(2001:3000,:)]; %2000x57
        CV_Train_Result_2 = [Y(1:1000, :) ; Y(2001:3000,:)]; %2000x1
        
        CV_Test_Set_2 = X(1001:2000, :);   %1000x57
        CV_Test_Result_2 = Y(1001:2000, :); %1000x1
        
        [w2 b2] = learnsvm2(CV_Train_Set_2,CV_Train_Result_2,C(c)); %57x1, 1x1

        R2 =  CV_Test_Set_2*w2 + b2; %1000x1
        
        error_count_2 = 0;
        for i = 1:1000
            if R2(i,1) >= 1 && CV_Test_Result_2(i,1) == -1
                error_count_2 = error_count_2 +1;
            elseif R2(i,1)< 1 && CV_Test_Result_2(i,1) == 1
                error_count_2 = error_count_2 +1;
            end
        end
        
        
        %3rd Fold
        CV_Train_Set_3 = X(1001:3000, :); %2000x57
        CV_Train_Result_3 = Y(1001:3000, :); %2000x1
        
        CV_Test_Set_3 = X(1:1000, :);   %1000x57
        CV_Test_Result_3 = Y(1:1000, :); %1000x1
        
        [w3 b3] = learnsvm2(CV_Train_Set_3,CV_Train_Result_3,C(c)); %57x1, 1x1
        
        
        R3 =  CV_Test_Set_3*w3 + b3; %1000x1
        
        error_count_3 = 0;
        for i = 1:1000
           if R3(i,1) >= 1 && CV_Test_Result_3(i,1) == -1
                error_count_3 = error_count_3 +1;
           elseif R3(i,1)< 1 && CV_Test_Result_3(i,1) == 1
               error_count_3 = error_count_3 +1;
           end
        end
        
        %Error rate calculation
        
        err_avg = (error_count_1 + error_count_2 + error_count_3)/(3*10);
        
        C_err(c) = err_avg;
        
        
        X_test = load('spamtrainX.data','-ascii'); %1601x57
        Y_test = load('spamtrainY.data','-ascii'); %1601x1
        [wt bt] = learnsvm2(X,Y,C(c));
    
        Rf =  X_test*wt + bt; %1601x1
        
        error_count_f = 0;
        for i = 1:1601
           if Rf(i,1) >= 1 && Y_test(i,1) == -1
                error_count_f = error_count_f +1;
           elseif Rf(i,1)< 1 && Y_test(i,1) == 1
               error_count_f = error_count_f +1;
           end
        end
        F_err(c) = (error_count_f/1601)*100;
    end
    hold on;
    semilogx(C,C_err,'b');
    semilogx(C,F_err,'r');
    legend('CrossValidation plot','Testing plot');
    hold off;
end