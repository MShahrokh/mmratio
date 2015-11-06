function Covariance = ShrinkageCov(Data,NumOfPoints)
SampleCovInit = cov(Data);
p = size(Data,2);
%%% shrinkage part
SampleCov = (NumOfPoints-1)/(NumOfPoints)*SampleCovInit;
rhoShrinkageHat = 0;

for kk = 1:NumOfPoints
    rhoShrinkageHat = rhoShrinkageHat + (norm((Data(kk,:))'*Data(kk,:)- SampleCov,'fro'))^2;
end
rhoShrinkageHat = rhoShrinkageHat/(NumOfPoints^2*(trace(SampleCov^2)-(trace(SampleCov))^2/p));
rhoShrinkageHat = min(rhoShrinkageHat,1);
%%% estimated shrinkage psi for class 0
Covariance = (1-rhoShrinkageHat)*SampleCov+rhoShrinkageHat*trace(SampleCov)/p*eye(p);

end
