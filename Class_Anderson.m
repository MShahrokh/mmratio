function Class = Class_Anderson(Point,Data0,Data1)

p = size(Data0,2);
n0 = size(Data0,1);
n1 = size(Data1,1);
% c = 0.5;
if n0 == 0
    Class = ones(size(Point,1),1);
elseif n1 == 0
    Class = zeros(size(Point,1),1);
else
    if n0 < p+1
        SigmaHat0 = ShrinkageCov(Data0,n0);%(cov(Data0)+0.001*eye(p))^(-1);
    else
        SigmaHat0 = cov(Data0);
    end
    if n1 < p+1
        SigmaHat1 = ShrinkageCov(Data1,n1);%(cov(Data0)+0.001*eye(p))^(-1);
    else
        SigmaHat1 = cov(Data1);
    end
    % SigmaHat1 = ShrinkageCov(Data1,n1);
    % SigmaHat0 = cov(Data0);
    % SigmaHat1 = cov(Data1);
    SigmaHat = ((n0-1)*SigmaHat0+(n1-1)*SigmaHat1)/(n0+n1-2);
    Mu0 = mean(Data0,1);
    Mu1 = mean(Data1,1);
    [Rw0 Col0] = size(Mu0);
    if Col0 > Rw0
        Mu0 = Mu0';
        Mu1 = Mu1';
    end
    
    SIGINV = SigmaHat^(-1); %(SigmaHat+0.001*eye(p))^(-1);
    a = SIGINV*(Mu1-Mu0);
    b = -1/2*(Mu1-Mu0)'*SIGINV*(Mu1+Mu0);%+log((1-c)/c);
    Discriminant = Point*a + b;
    Class = Discriminant>0;
end
end

