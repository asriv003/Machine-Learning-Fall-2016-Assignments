function main()
%
%
%
    clc
    load -ascii handwriting.data;
    Y = handwriting(:,1);
    X = handwriting(:,2:end);
    %for i = 1:25
    %    subplot(5,5,i)
    %    imagesc(reshape(X(i,2:end),[8 16])');
    %    colormap (1.0 - gray);
    %    axis equal;
    %end
end