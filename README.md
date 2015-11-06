# mmratio
minimax re-sampling analysis
The mmratio function is a MATLAB implementation of Algorithm 1 detailed in the paper entitled: "Effect of Separate Sampling on Classification Accuracy" by Esfahani and Dougherty, Bioinformatics, 2014. The function takes a
number of inputs whereas two of them are necessary: 

1- Sample data in the form a matrix whose rows and columns indicate samples and feature vector, respectively. Therefore, sample data is processed as an n × D matrix. 

2- Vector of labels corresponding to each row in the sample data. We consider a binary-classification problem, and hence, it should only contain two types of numeric labels. In the case that labels are different from 0 and 1, 

* We define the ratio r as the number of sample size with the smaller numeric label to the total sample size.

* Other inputs are optional but can highly affect the result. 

* The default classification rule is 3NN. 

* The default number of initial feature selection recommended for kNN rules, is 1000. The following functions are also needed for mmratio to run: Class_LDA, Class_QDA, Class_KNN, Class_Anderson, Class_NM, LibSVM package, and MATLAB classregtree.
