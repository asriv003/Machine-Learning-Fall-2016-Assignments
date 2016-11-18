function q1()
%Abhishek Kumar Srivastava
%Student Id: 861307778
%October 12, 2016
%CS 229
%PS2
%Solution of Q1

    %clear up workspaces
    clc;
    f1 = figure;	%for part a and part c
    f2 = figure;	%for part b
    figure(f1);
    hold on
    %load file into matrix
    D = load('comm.txt','-ascii');

    %extracting training data
    TrainData = D(1:1000,1:99);     %size 1000x99
    TrainResult = D(1:1000,100);    %size 1000x1

    %extracting test data
    TestData = D(1001:1994,1:99);   %size 994x99
    TestResult = D(1001:1994,100);  %size 994x1

    %part a

    %lasso regression on the training set 
    %λ values from 10^−6 to 10^−1 taking 100 random Values
    LambdaValues = logspace(-6,-1,100);
    %sorting randomly selected data
    LambdaValues = sort(LambdaValues);
    %lasso training
    [TrainWt TrainInfo] = lasso(TrainData,TrainResult,'Lambda',LambdaValues);
    %multiplying weights to the testing predictors
    TrainAns = TrainData*TrainWt;
    
    %computing the mean average squared error on the training data
    train_avg = zeros(size(LambdaValues));
    for l = 1:size(LambdaValues,2)
        avg = 0;
        for i=1:1000
            avg = avg + mean((TrainResult(i,:)-TrainAns(i,l)).^2);
        end
        train_avg(l) = avg/1000;
    end
    %ploting graph average sq error for training data vs lambda
    semilogx(LambdaValues,train_avg,'r');
    
    %computing the average squared error on the test data
    TestAns = TestData*TrainWt;
    test_avg = zeros(size(LambdaValues));
    for l = 1:size(LambdaValues,2)
        avg = 0;
        for i=1:994
            avg = avg + mean((TestResult(i,:)-TestAns(i,l)).^2);
        end
        test_avg(l) = avg/1000;
    end
    %plotting graph average sq error for testing data vs lambda
    semilogx(LambdaValues,test_avg,'b');
    
    xlabel('Lambda');
    ylabel('Average Squared Error');
    title('Part A & Part C');
    hold off
    
    %part b
    %plotting graph between the weights and λ
    figure(f2);
    semilogx(LambdaValues,TrainWt);
    xlabel('Lambda');
    ylabel('Weights');
    title('Part B');
    
    %part c
    %cross validation
    figure(f1);
    hold on
    %10 folds
    RowVal = 1:100:1000;    %[1 101 201 ... 901]
    
    CV_D = D(1:1000,:);     %Data for cross validation i.e Training data
    
    cv_avg_final = zeros(size(LambdaValues));
    
    %dividing Testing data in 10 parts
    %first we use 10th part as testing data then 9th and so on.
    for fold = 1:10
      %CV Test Set from Training Data
      CVTestSet = CV_D(RowVal(11-fold):(RowVal(11-fold)+99),:); %100x100
      %CV Training Set after removing CV Test Set from Original Training data 
      CVTrainSet = setdiff(CV_D,CVTestSet,'rows');     %900x100
      %lasso training on CV Training Set
      [CVWeight,CVInfo] = lasso(CVTrainSet(:,1:99),CVTrainSet(:,100),'Lambda',LambdaValues);
      
      CVAns = CVTestSet(:,1:99)*CVWeight;  
      
      cv_avg = zeros(size(LambdaValues));
      %calculating average for each lambda
      for l = 1:size(LambdaValues,2)
        avg = 0;
        for i=1:100
            avg = avg + ((CVTestSet(i,100)-CVAns(i,l)).^2);
        end
        cv_avg(l) = avg/100;
      end
      
      cv_avg_final = cv_avg + cv_avg_final;
    end
    
    cv_avg_final = cv_avg_final./10;
    %plotting data average sq error for Cross Validation vs lambda
    semilogx(LambdaValues,cv_avg_final,'k');
    
    legend('Train plot','Test plot', 'Cross Validation');
    hold off
end
