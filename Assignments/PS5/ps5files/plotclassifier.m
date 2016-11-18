function plotclassifier(trainX,trainY,f,Ythresh,marginval)

if nargin<4
	Ythresh = 0;
end
if nargin<5
	marginval = 1;
end
h1 = plot(trainX(trainY<Ythresh,1),trainX(trainY<Ythresh,2),...
				'ro','markersize',9,'linewidth',2);
hstate = ishold;
hold on;
h2 = plot(trainX(trainY>Ythresh,1),trainX(trainY>Ythresh,2),...
				'bx','markersize',9,'linewidth',2);
axis equal;
a = axis;
[xs,ys] = meshgrid(linspace(a(1),a(2),50),linspace(a(3),a(4),50));
X = [reshape(xs,prod(size(xs)),1) reshape(ys,prod(size(ys)),1)];
F = f(X);
fs = reshape(F,size(xs));
if exist('fcontour')
	if marginval
		caxis([Ythresh-marginval*2 Ythresh+marginval*2]);
		colormap([1 0.625 0.625; 0.875 0.625 0.75; 0.75 0.625 0.875; 0.625 0.625 1]);
		contourf(xs,ys,fs,[-inf Ythresh-marginval Ythresh Ythresh+marginval]);
	else
		caxis([-2 2]);
		colormap([0 0 0; 1 0.625 0.625; 0.625 0.625 1]);
		[c,h] = contourf(xs,ys,fs,[-inf Ythresh +inf]);
	end
else
	if marginval
		caxis([Ythresh-marginval*2 Ythresh+marginval*2]);
		colormap([1 0.625 0.625; 0.875 0.625 0.75; 0.75 0.625 0.875; 0.625 0.625 1]);
		contourf(xs,ys,fs,[-inf Ythresh-marginval Ythresh Ythresh+marginval]);
	else
		caxis([-2 2]);
		colormap([1 0.625 0.625; 0.625 0.625 1]);
		contourf(xs,ys,fs,[-inf Ythresh]);
	end
end
[c,h] = contour(xs,ys,fs,[Ythresh Ythresh]);
h.LineColor = [0 0 0];
h.LineWidth = 2;
uistack([h1 h2],'top');
if ~ishold
	hold off;
end
