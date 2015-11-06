function Error = Hold_Out_Error(Classifier,Features,Labels)

% Classifier = @(Point) Class_OBC(Point,Expected_C,Mu0, Mu1,Psi0, Psi1, Nu0,Nu1);
NumOfPoint = length(Labels);
% % L0 = Labels(Labels==0);
% % L1 = Labels(Labels==1);
% % LenL0 = length(L0);
% % LenL1 = length(L1);
ClassifiedLabels = Classifier(Features);
% % CL0 = ClassifiedLabels(1:LenL0);
% % CL1 = ClassifiedLabels(LenL0+1:LenL1+LenL0);
% size(ClassifiedLabels)
% size(Labels)
% size(CL0)
% size(L0)
Error = sum(ClassifiedLabels~=Labels)/(NumOfPoint);
% ERR1 = sum(CL1~=L1);
% Error = ERR0/(LenL1+LenL0) + ERR1/(LenL1+LenL0);
end

