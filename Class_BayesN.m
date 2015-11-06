function Class = Class_BayesN(Point,c,CovMat0,Mu0,CovMat1,Mu1)
% p = length(Point);
CovMatInv0 = CovMat0^(-1);
CovMatInv1 = CovMat1^(-1);
A = -1/2*(CovMatInv1-CovMatInv0);
b = CovMatInv1*Mu1 - CovMatInv0*Mu0;
D = -1/2*(Mu1'*CovMatInv1*Mu1-Mu0'*CovMatInv0*Mu0)-1/2*log(det(CovMat1)/det(CovMat0))+log((1-c)/(c));
% size(Point)
% size(A)
% size(b)
% size(c)
% NumOfPoints = size(Point,1);
% b = repmat(b',NumOfPoints,1);
% size(b)
Discriminant = sum((Point*A).*Point,2)+ Point*b + D;
Class = Discriminant>0;
end

