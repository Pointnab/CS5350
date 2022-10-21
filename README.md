# CS5350
This is a machine learning library developed by Ryan Lam for CS5350 in University of Utah

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
