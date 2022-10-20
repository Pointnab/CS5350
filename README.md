# CS5350
This is a machine learning library developed by Ryan Lam for CS5350 in University of Utah

In order to learn a decision tree, import the InformationGain, MajorityError, or GiniIndex classes as needed

Create a tree object using the classes' respective constructor 
(i.e. InformationGain.InfoGain(), MajorityError.ME(), and GiniIndex.GI())

call the object method train() with the following parameters:
training data - pandas dataframe
depth (optional) - an integer for the maximum number of times to split the data, default:6
binary (optional) - TRUE to convert numerical features into binary features using the media as a threshold, default:FALSE
unknown (optional) - TRUE to treat "Unknown" as a missing value instead of a unique category, default:FALSE