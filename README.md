# CS5350
This is a machine learning library developed by Ryan Lam for CS5350 in University of Utah
The SVM algorithm methods are run through a test file with command line arguments as following:
SVM.py <file path to training data> <file path to test data> <SVM mode> <a or g>
where mode can be:
"P" for primal SVM
"D" for dual SVM (dual SVM is broken and reports nonsense)
"K" for dual SVM with gaussian kernel (this is adversely affected by the prior mode being broken)
If the mode is "P", the fourth argument specifies a for the learning rate schedule, a value of 0 uses the simpler schedule
If the mode is "K", the fourth argument speecifies gamma for the the gaussian kernel
The test will train each model and report the weight vector and bias for the 3 values of C.

The perceptron methods are run through a test file with command line arguments as following:
Perceptron.py <file path to training data> <file path to test data> <perceptron mode>
where mode can be:
"S" for standard perceptron
"V" for voted perceptron
"A" for averaged perceptron
Test will train, test, and reset the model 100 times. Reporting the average test error and the final weights vector

Do not run the run.sh file in the ensemble learning folder, code is too slow
classes are run through a corresponding test file with command line arguments as following:
AdaBoostTest.py <file path to training data> <file path to test data> <# of iterations>
BagTest.py <file path to training data> <file path to test data> <# of trees> <# of samples per tree>
ForestTest.py <file path to training data> <file path to test data> <# of trees> <# of samples per tree>
  *ForestTest.py tests feature subset sizes of 2, 4, and 6 automatically.
Do not run the bias and variance scripts, 100 models with 500 trees each takes too long
  
The Linear Regrssion code takes the following:
BatchTest.py <file path to training data> <file path to test data> <# of iterations>
SGDTest.py <file path to training data> <file path to test data> <# of iterations>
  *for both batch and stochastic, the learning rate is set in the testing code

In order to learn a decision tree, import the InformationGain, MajorityError, or GiniIndex classes as needed

Create a tree object using the classes' respective constructor 
(i.e. InformationGain.InfoGain(), MajorityError.ME(), and GiniIndex.GI())

call the object method train() with the following parameters:
training data - pandas dataframe
depth (optional) - an integer for the maximum number of times to split the data, default:6
binary (optional) - TRUE to convert numerical features into binary features using the media as a threshold, default:FALSE
unknown (optional) - TRUE to treat "Unknown" as a missing value instead of a unique category, default:FALSE
