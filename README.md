# Credit_Risk_Customer_Prediction

## Lab1: Visualizing Data for Classification

In this lab, your goal is to explore a dataset that includes information about German bank credit to understand the relationships for a classification problem. In classification problems the label is a categorical variable.

Colinear features should be identified so they can be eliminated or otherwise dealt with. However, for classification problems you are looking for features that help separate the label categories. Separation is achieved when there are distinctive feature values for each label category. Good separation results in low classification error rate.

### By the completion of this lab, you will:

1. Examine the imbalance in the label cases using a frequency table.      
2. Find numeric or categorical features that separate the cases using visualization.

### Lab Steps

1. Make sure that you have completed the setup requirements as described in requirement.txt.
2. Now, run jupyter notebook and open the “VisualizingDataForClassification.ipynb” notebook under this project.
3. Examine the notebook and answer the questions along the way.

**Question1:** From the created plots, which two features seem to separate the good and bad credits?      
**Question2:** From the created plots, which feature seems to separate the good and bad credits?

## Lab2: Data Preparation

**Data preparation** is a vital step in the machine learning pipeline. Just as visualization is necessary to understand the relationships in data, proper preparation or **data munging** is required to ensure machine learning models work optimally.

The process of data preparation is highly interactive and iterative. A typical process includes at least the following steps:

1. **Visualization** of the dataset to understand the relationships and identify possible problems with the data.
2. **Data cleaning** and transformation to address the problems identified. It many cases, step 1 is then repeated to verify that the cleaning and transformation had the desired effect.
3. **Construction and evaluation of a machine learning models.** Visualization of the results will often lead to understanding of further data preparation that is required; going back to step 1.

### By the completion of this lab, you will:

1. Recode character strings to eliminate characters that will not be processed correctly.
2. Find and treat missing values.
3. Set correct data type of each column.
4. Transform categorical features to create categories with more cases and likely to be useful in predicting the label.
5. Apply transformations to numeric features and the label to improve the distribution properties.
6. Locate and treat duplicate cases.

### Lab Steps

1. Make sure that you have completed the setup requirements as described in requirement.txt.
2. Now, run jupyter notebook and open the “DataPreparation.ipynb” notebook under this project.
3. Examine the notebook and answer the questions along the way.

**Question1:** How many cases have duplicates?

## Lab3: Classification

In this lab you will perform **two-class classification** using **logistic regression**. A classifier is a machine learning model that separates the **label** into categories or **classes**. In other words, classification models are supervised machine learning models which predict a categorical label.

The German Credit bank customer data is used to determine if a particular person is a good or bad credit risk. Thus, credit risk of the customer is the classes you must predict. In this case, the cost to the bank of issuing a loan to a bad risk customer is five times that of denying a loan to a good customer. This fact will become important when evaluating the performance of the model.

### By the completion of this lab, you will:

1. Prepare data for classification models using scikit-learn.
2. Construct a classification model using scikit-learn.
3. Evaluate the performance of the classification model.
4. Use techniques such as reweighting the labels and changing the decision threshold to change the trade-off between false positive and false negative error rates.

### Lab Steps

1. Make sure that you have completed the setup requirements as described in requirement.txt.
2. Now, run jupyter notebook and open the “Classification.ipynb” notebook under this project.
3. Examine the notebook and answer the questions along the way.

**Question1.** Knowing the class imbalances in the data, theoretically, what is the best accuracy you can get without creating any machine learning models?   
**Question2:** During the one-hot encoding process, the six categorical features were converted to 31 dummy variables. How many dummy variables came from the checking_account_status feature?   
**Question3:** What is the AUC of the model?   
**Question4:** What three metrics may change by giving weights to the classes?

## Lab4: Cross Validation

Cross validation is a widely used resampling method. It repeats a calculation multiple times using randomly selected subsets of the complete dataset.

To obtain unbiased estimates of expected model performance while performing model selection, it is necessary to use nested cross validation. As the name implies, nested cross validation is performed though a pair of nested CV loops. The outer loop uses a set of folds to perform model evaluation. The inner loop performs model selection using another randomly sampled set of folds not used for evalution by the outer loop. This algorithm allows model selection and evaluation to proceed with randomly sampled subsets of the full data set, thereby avoiding model selection bias.

In this lab you will perform simple cross validation and nested cross validation.

### Lab Steps

1. Make sure that you have completed the setup requirements as described in requirement.txt.
2. Now, run jupyter notebook and open the “CrossValidation.ipynb” notebook under this project.
3. Examine the notebook and answer the questions along the way.

**Question1:** Which fold has the highest AUC?   
**Question2:** What is the mean performance metric of the cross-validated model?

## Lab5: Feature Selection

**Feature selection** can be an important part of model selection. In supervised learning, including features in a model which do not provide information on the label is useless at best and may prevent generalization at worst.

Feature selection can involve application of several methods. Two important methods include:

1. Eliminating features with **low variance** and **zero variance**. Zero variance features are comprised of the same values. Low variance features arise from features with most values the same and with few unique values. One way low variance features can arise, is from dummy variables for categories with very few members. The dummy variable will be mostly 0s with very few 1s.
2. Training machine learning models with features that are **uninformative** can create a variety of problems. An uniformative feature does not significantly improve model performance. In many cases, the noise in the uninformative features will increase the variance of the model predictions. In other words, uniformative models are likely to reduce the ability of the machine learning model to generalize.

### By the completion of this lab, you will:

1. Eliminate low variance features, which by their nature cannot be highly informative since they contain a high fraction of the same value.
2. Use recursive feature elimination, a cross validation technique for identifying uninformative features.

### Lab Steps

1. Make sure that you have completed the setup requirements as described in requirement.txt.
2. Now, run jupyter notebook and open the “FeatureSelection.ipynb” notebook under this project.
3. Examine the notebook and answer the questions along the way.

**Question1:** How many features are deemed low variance?    
**Question2:** What is the AUC of the model?      

## Lab6: Dimensionality Reduction

**Principle component analysis**, or **PCA**, is an alternative to regularization and stright-forward feature elimination. PCA is particularly useful for problems with very large numbers of features compared to the number of training cases. For example, when faced with a problem with many thousands of features and perhaps a few thousand cases, PCA can be a good choice to **reduce the dimensionality** of the feature space.

### By completion of this lab, you will:

1. Compute PCA models with different numbers of components.
2. Compare logistic regression models with different numbers of components.

### Lab Steps

1. Make sure that you have completed the setup requirements as described in requirement.txt.
2. Now, run jupyter notebook and open the “DimensionalityReduction.ipynb” notebook under this project.
3. Examine the notebook and answer the questions along the way.

**Question1:** What is the AUC of the model with 5 components?       
**Question2:** What is the AUC of the model with 10 components?     

## Lab7: Bagging

### Lab Steps

1. Make sure that you have completed the setup requirements as described in requirement.txt.
2. Now, run jupyter notebook and open the “Bagging.ipynb” notebook under this project.
3. Examine the notebook and answer the questions along the way.
     
**Question1:** What is the best value of max_features?     
**Question2:** What is the AUC of the model?      

## Lab8: Boosting

### Lab Steps

1. Make sure that you have completed the setup requirements as described in requirement.txt.
2. Now, run jupyter notebook and open the “Boosting.ipynb” notebook under this project.
3. Examine the notebook and answer the questions along the way.

**Question1:** What is the best value of learning_rate?     
**Question2:** What is the best value of learning_rate?    
**Question3:** What is the AUC of the model?    

## Lab9: Neural Networks

### Lab Steps

1. Make sure that you have completed the setup requirements as described in requirement.txt.
2. Now, run jupyter notebook and open the “NeuralNetworks.ipynb” notebook under this project.
3. Examine the notebook and answer the questions along the way.
    
**Question1:** What is the best value of beta_1?       
**Question2:** What is the AUC of the model?       

## Lab10: SVM

### Lab Steps

1. Make sure that you have completed the setup requirements as described in requirement.txt.
2. Now, run jupyter notebook and open the “SupportedVectorMachines.ipynb” notebook under this project.
3. Examine the notebook and answer the questions along the way.

**Question1:** What is the best value of gamma?      
**Question2:** What is the AUC of the model?       

## Lab11: Naive Bayes

### Lab Steps

1. Make sure that you have completed the setup requirements as described in requirement.txt.
2. Now, run jupyter notebook and open the “NaiveBayes.ipynb” notebook under Module 6 folder.
3. Examine the notebook and answer the questions along the way.

**Question1:** What is the AUC of the Gaussian Naive Bayes model?      
**Question2:** What is the AUC of the Bernoulli Naive Bayes model?       





