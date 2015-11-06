function Class = Class_KNN(Point,Data0,Data1,K)
N0 = size(Data0,1);
N1 = size(Data1,1);
N = N0 + N1;
% DIME = size(Data0,2);
NumberOfPoints = size(Point,3);
%%%% the variable Point is the points to be classified; it is a 3D matrix
%%%% where the first dimension is as the number of TRAINING size, the
%%%% second is FEATURE size, and the third is the NUMBER of points to be
%%%% classified
DATAAgreggation = [Data0;Data1];
TrainingAgregg = repmat(DATAAgreggation,[1,1,NumberOfPoints]);
% % size(Point)
% % size(TrainingAgregg)
DISTANCEElements = sum((TrainingAgregg - Point).^2,2);
DISTANCEElements(1:N0,:,:) = DISTANCEElements(1:N0,:,:);
DISTANCEElements(N0+1:N,:,:) = (-1)*DISTANCEElements(N0+1:N,:,:);
DISTANCEElements = squeeze(DISTANCEElements);
% % size(DISTANCEElements)
[~,SORTEDElementsid] = sort(abs(DISTANCEElements));
DISTANCEElementsNew = DISTANCEElements(SORTEDElementsid);
SortedLabels = sign(DISTANCEElementsNew);
SelectedSortedLabels = SortedLabels(1:K,:);
% N
% size(sum(SelectedSortedLabels,1))
MajorityVoteCount(1:NumberOfPoints,1) = sum(SelectedSortedLabels,1);
Class = MajorityVoteCount<=0;

end

