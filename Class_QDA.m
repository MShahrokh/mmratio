function Class = Class_QDA(Point,Data0,Data1)
p = size(Point,2);
NumOfPoints0 = size(Data0,1);
NumOfPoints1 = size(Data1,1);
c = NumOfPoints0/(NumOfPoints0+NumOfPoints1);
if NumOfPoints0 < p+1
    SigmaHat0Inv = (ShrinkageCov(Data0,NumOfPoints0))^(-1);%(cov(Data0)+0.001*eye(p))^(-1);
else
    SigmaHat0Inv = (cov(Data0))^(-1);
end
if NumOfPoints1 < p+1
    SigmaHat1Inv = (ShrinkageCov(Data1,NumOfPoints1))^(-1);%(cov(Data1)+0.001*eye(p))^(-1);
else
    SigmaHat1Inv = (cov(Data1))^(-1);
end

Mu0 = mean(Data0,1);
Mu1 = mean(Data1,1);
[Rw0 Col0] = size(Mu0);
if Col0 > Rw0
    Mu0 = Mu0';
    Mu1 = Mu1';
end
A = -1/2*(SigmaHat1Inv-SigmaHat0Inv);
b = SigmaHat1Inv*Mu1 - SigmaHat0Inv*Mu0;
C = -1/2*(Mu1'*SigmaHat1Inv*Mu1-Mu0'*SigmaHat0Inv*Mu0)-1/2*log(det(SigmaHat0Inv)/det(SigmaHat1Inv))+log((1-c)/c);
Discriminant = sum((Point*A).*Point,2)+ Point*b +C;
Class = Discriminant>0;
end

