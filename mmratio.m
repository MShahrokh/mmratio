% this is a function impelementing the algorithm for approximation of the
% minimax ratio for a given dataset and user-defined classification rule,
% number of final features to be selected, any intial feature selection,
% test size for two classes in order to do holdout, maximum number of
% iterations for expected holdout error estimation
% the ratio is defined as the number of sample points with a smaller
% numberic label representation to the total sample size.
function MinMaxRatio = mmratio(sample_data, labels, rule_name,initial_fs,final_fs,test_size_0,test_size_1,max_iter,varargin)
D = size(sample_data,2);

if nargin < 8
    max_iter = 2000;
    if nargin < 7
        test_size_0 = 30;
        if nargin < 6
            test_size_1 = 30;
            if nargin < 5
                final_fs = 15;
                if nargin < 4
                    initial_fs = min(500,D);
                    if nargin < 3
                        rule_name = {'3NN'};
                    end
                end
            end
        end
    end
end
if isempty(rule_name)
    rule_name = {'3NN'};
end
if isempty(initial_fs)
    initial_fs = min(500,D);
end
if isempty(final_fs)
    final_fs = 15;
end
if isempty(test_size_0)
    test_size_0 = 30;
end
if isempty(test_size_1)
    test_size_1 = 30;
end
if isempty(max_iter)
    max_iter = 2000;
end

% we highly recommend initial feature selection when deal with KNN
% classification rules

if nargin < 2
    error('This program needs to have at least two inputs, one for the dataset and one labels')
end
p = final_fs;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lable_Numbers = unique(labels);
if length(Lable_Numbers) > 2
    error('This program is only written for a binary-classification problem')
end
labels(labels==Lable_Numbers(1)) = 0;
labels(labels==Lable_Numbers(2)) = 1;
Data_0 = sample_data(labels==0,:);
Data_1 = sample_data(labels==1,:);
[~,InitialFeatures] = TTest(Data_0,Data_1,initial_fs);
Data_0 = Data_0(:,InitialFeatures);
Data_1 = Data_1(:,InitialFeatures);
MaxIterMC = max_iter;
NumOfPoints_0 = size(Data_0,1);
NumOfPoints_1 = size(Data_1,1);
Test_Size_0 = test_size_0;
Test_Size_1 = test_size_1;
TotalSampleSize = min(NumOfPoints_0-Test_Size_0,NumOfPoints_1-Test_Size_1);
NumOfPoints = TotalSampleSize;
Portions = 1/NumOfPoints:1/NumOfPoints:1-1/NumOfPoints; % different r values
Classifiers = {'LDA' 'QDA' '3NN' '5NN' 'L-SVM' 'DT' 'Anderson' 'RBF-SVM' 'NM'};
warning off

Classifiers_Studied = rule_name;%{'3NN' '5NN' 'SVM' 'DecisionTree' 'RBF-SVM'};
if iscell(Classifiers_Studied)
    MaxClssRules = length(Classifiers_Studied);
else
    MaxClssRules = 1;
end
mismatch = 0;
active_flag = zeros(1,MaxClssRules);

for ii = 1:MaxClssRules
    if iscell(Classifiers_Studied)
        if isempty(find(strcmpi(Classifiers, Classifiers_Studied{ii}),1))
            mismatch = mismatch +1;
        else
            active_flag(ii) = 1;
        end
    else
        if isempty(find(strcmpi(Classifiers, Classifiers_Studied),1))
            mismatch = mismatch +1;
        else
            active_flag(ii) = 1;
        end
    end
end

if mismatch == MaxClssRules
    error('The classifer names are not in the allowed ones');
end
if iscell(Classifiers_Studied)
    MaxClssRules = length(Classifiers_Studied);
else
    MaxClssRules = 1;
end

r_minimax = zeros(1,MaxClssRules);
for rr = 1:MaxClssRules
    
    if active_flag(rr) == 1
        if iscell(Classifiers_Studied)
        ClassRuleName = Classifiers_Studied{rr};
        else
            ClassRuleName = Classifiers_Studied;
        end
        SplitStrategy = Portions;
        jj = 0;
        Slope_Means_New = 1;
        ERRORS = zeros(MaxIterMC,2);
        ERRORS_Means = zeros(1,2);
        AA = zeros(9,1);
        SIGN = 1;
        while SIGN > 0 && jj < NumOfPoints - 1 
            jj = jj + 1;
            r = SplitStrategy(jj);
            Slope_Means_Old = Slope_Means_New;
            N0 = round(TotalSampleSize*(SplitStrategy(jj)));
            N1 = TotalSampleSize - N0;
            parfor ii = 1:MaxIterMC
                Index0 = randsample(NumOfPoints_0,N0);
                Index1 = randsample(NumOfPoints_1,N1);
                ERRORS(ii,:) = GiveMeSlopeHoldOut(Data_0,Data_1,Index0,Index1,N0,N1,p,ClassRuleName);
            end
            ERRORS_Means = mean(ERRORS,1);
            Slope_Means = ERRORS_Means(1) - ERRORS_Means(2);
            Slope_Means_New = Slope_Means;
            AA = ERRORS_Means(1);
            SIGN = Slope_Means_New*Slope_Means_Old;
            
        end
        
        if SIGN == 0
            r_minimax(rr) = r;
        elseif AA < ERRORS_Means(2)
            r_minimax(rr) = r - 1/TotalSampleSize;
        elseif AA > ERRORS_Means(2)
            r_minimax(rr) = r;
        elseif AA == ERRORS_Means(2)
            r_minimax(rr) = r;
        else
            r_minimax(rr) = 1;
        end
    else
        warning('MATLAB:paramAmbiguous','The %d-th classification rule %s is not allowed',rr,ClassRuleName)
        r_minimax(rr) = NaN;
    end
    
end
MinMaxRatio = r_minimax;
end
%%%%%Here is another function utilized in mmratio
function Slope = GiveMeSlopeHoldOut(Data_0,Data_1,Index0,Index1,Nt_0,Nt_1,p,RuleName)
% Slope = zeros(2,1);
TotalSampleSize = Nt_0 + Nt_1;
Data0 = Data_0(Index0,:);
Data1 = Data_1(Index1,:);
TRank = p;

FlagLda = find(strcmpi('LDA', RuleName));
FlagQda = find(strcmpi('QDA', RuleName));
Flag3NN = find(strcmpi('3NN', RuleName));
Flag5NN = find(strcmpi('5NN', RuleName));
FlagDT = find(strcmpi('DT', RuleName));
FlagSVM = find(strcmpi('L-SVM', RuleName));
FlagKSVM = find(strcmpi('RBF-SVM', RuleName));
FlagAnder = find(strcmpi('Anderson', RuleName));
FlagNM = find(strcmpi('NM', RuleName));
[~,SelectedFeatures] = TTest(Data_0,Data_1,TRank);
Data0 = Data0(:,SelectedFeatures);
Data1 = Data1(:,SelectedFeatures);
RemainingIndex0 = setdiff(1:size(Data_0,1),Index0);

RemainingIndex1 = setdiff(1:size(Data_1,1),Index1);
TestSize0 = length(RemainingIndex0);
TestSize1 = length(RemainingIndex1);


NumOfPoint = TestSize0 + TestSize1;

DataTest0 = Data_0(RemainingIndex0,SelectedFeatures);
DataTest1 = Data_1(RemainingIndex1,SelectedFeatures);
Features = [DataTest0;DataTest1];
% % % % % %

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
AggreGateData = [Data0;Data1];
n0 = size(Data0,1);
n1 = size(Data1,1);
%%%% working with labels; used for all the classification rules
Labels(1:TestSize0,1) = 0;
Labels(TestSize0+1:TestSize0+TestSize1,1) = 1;
LABELS(1:n0,1) = 0;
LABELS(n0+1:n0+n1,1) = 1;
L0 = Labels(Labels==0);
L1 = Labels(Labels==1);
LenL0 = length(L0);
LenL1 = length(L1);

if FlagLda == 1
    LDA_Classifier = @(x) Class_LDA(x,Data0,Data1);
    Slope = Hold_Out_Slope(LDA_Classifier,Features,Labels);
elseif FlagQda == 1
    QDA_Classifier = @(x) Class_QDA(x,Data0,Data1);
    Slope = Hold_Out_Slope(QDA_Classifier,Features,Labels);
elseif Flag3NN == 1
    FeaturesKNN = zeros(TotalSampleSize,p,NumOfPoint);
    for ii = 1:TotalSampleSize
        FeaturesKNN(ii,:,:) = Features';
    end
    K3NN_Classifier = @(x) Class_KNN(x,Data0,Data1,3);
    Slope = Hold_Out_Slope(K3NN_Classifier,FeaturesKNN,Labels);
    %     Slope
elseif Flag5NN == 1
    FeaturesKNN = zeros(TotalSampleSize,p,NumOfPoint);
    for ii = 1:TotalSampleSize
        FeaturesKNN(ii,:,:) = Features';
    end
    K5NN_Classifier = @(x) Class_KNN(x,Data0,Data1,5);
    Slope = Hold_Out_Slope(K5NN_Classifier,FeaturesKNN,Labels);
    
elseif FlagDT == 1
    CategoriZ(1:n0,1) = 0;
    CategoriZ(n0+1:n0+n1,1) = 1;
    ConstructedTree = classregtree(AggreGateData,CategoriZ,'method','classification');
    [~,~,Predicted_Labels] = eval(ConstructedTree,Features);
    Predicted_Labels = Predicted_Labels - 1;
    ErroR0 = sum(Predicted_Labels(1:LenL0)~=L0)/(LenL0);
    ErroR1 = sum(Predicted_Labels(LenL0+1:LenL0+LenL1)~=L1)/(LenL1);
    if n0 == 0
        Slope = [1 0];
    elseif n1 == 0
        Slope = [0 1];
    else
        Slope = [ErroR0 ErroR1];
    end
elseif FlagSVM == 1
    SVMModel = svmtrain(LABELS, AggreGateData,'-t 0 -q');
    [Predicted_Labels] = svmpredict(Labels, Features, SVMModel,'-q');
    ErroR0 = sum(Predicted_Labels(1:LenL0)~=L0)/(LenL0);
    ErroR1 = sum(Predicted_Labels(LenL0+1:LenL0+LenL1)~=L1)/(LenL1);
    Slope = [ErroR0 ErroR1];
elseif FlagKSVM == 1
    SVMModelKernel = svmtrain(LABELS, AggreGateData,'-t 2 -q');
    [Predicted_Labels_Kernel] = svmpredict(Labels, Features, SVMModelKernel,'-q');
    ErroR0 = sum(Predicted_Labels_Kernel(1:LenL0)~=L0)/(LenL0);
    ErroR1 = sum(Predicted_Labels_Kernel(LenL0+1:LenL0+LenL1)~=L1)/(LenL1);
    Slope = [ErroR0 ErroR1];
elseif FlagAnder == 1
    Anderson_Classifier = @(x) Class_Anderson(x,Data0,Data1);
    Slope = Hold_Out_Slope(Anderson_Classifier,Features,Labels);
elseif FlagNM == 1
    NM_Classifier = @(x) Class_NM(x,Data0,Data1);
    Slope = Hold_Out_Slope(NM_Classifier,Features,Labels);
end

end
%%%%%%%
function Errs = Hold_Out_Slope(Classifier,Features,Labels)

% Classifier = @(Point) Class_OBC(Point,Expected_C,Mu0, Mu1,Psi0, Psi1, Nu0,Nu1);
NumOfPoint = length(Labels);
L0 = Labels(Labels==0);
L1 = Labels(Labels==1);
LenL0 = length(L0);
LenL1 = length(L1);
ClassifiedLabels = Classifier(Features);
CL0 = ClassifiedLabels(1:LenL0);
CL1 = ClassifiedLabels(LenL0+1:LenL1+LenL0);
% size(ClassifiedLabels)
% size(Labels)
% size(CL0)
% size(L0)
ERR0 = sum(CL0~=L0)/(LenL0);
ERR1 = sum(CL1~=L1)/(LenL1);
% [ERR0 ERR1]s
% Error = sum(ClassifiedLabels~=Labels)/(NumOfPoint);
% ERR1 = sum(CL1~=L1);
% Error = ERR0 - ERR1;

Errs = [ERR0 ERR1];
end

