function q1(numrep)
%Abhishek Kumar Srivastava
%Student Id: 861307778
%November 8, 2016
%CS 229
%PS5
%Question 1
    I = load('class2d.ascii','-ascii');
    %input values
    B = ones(80,1);
    X = I(:,1:2);   %80x2
    Y = I(:,3);     %80x1
	%Layers = 2, Hidden units = 1
    l2_h1(1,numrep,B,X,Y);
    %Layers = 2, Hidden units = 5
	l2_h5(2,numrep,B,X,Y);
    %Layers = 2, Hidden units = 20
	l2_h20(3,numrep,B,X,Y);
    %Layers = 3, Hidden units = 1
	l3_h1(4,numrep,B,X,Y);
    %Layers = 3, Hidden units = 5
	l3_h5(5,numrep,B,X,Y);
    %Layers = 3, Hidden units = 20
	l3_h20(6,numrep,B,X,Y);
end