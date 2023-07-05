# Credit Risk Prediction

Credit risk prediction is an effective way to know about the possibility of a loss for a lender due to a borrower’s failure to repay a loan.
This project is a final task of ID/X Parter x Rakamin Academy - Data Scientist Virtual Internship Experience Program.

### Dataset

The dataset we’re using can be found on Kaggle and it contains 466285 rows and 75 columns.
Data source: https://www.kaggle.com/datasets/devanshi23/loan-data-2007-2014

### Define Target Variable
Target Variable is the feature of a dataset about which we want to gain a deeper understanding and learn patterns between the dataset.
In this dataset, we can use the loan_status column to see the ending of each loans.

### Data Exploration and Preprocessing
In this step, we can check the dataset for missing values, duplicate, inconsistent columns and etc.

Drop identities columns (which have each unique values but irrelevant)<br>
Drop columns that contain >50% missing values.<br>
Drop columns that only have 1 unique values.<br>
Drop columns that contain data leakage.<br>
Drop columns that have similar values.

### One Hot Encoding
One hot encoding is a technique that we use to represent categorical variables as numerical values in a machine learning model.
This technique can improve model performance by providing more information to the model about the categorical variable.

### Standardization
Standardization entails scaling data to fit a standard normal distribution.
A standard normal distribution is defined as a distribution with a mean of 0 and a standard deviation of 1

### Data Splitting
We can split the data to easily evaluate the performance of our model. Such as, if it performs well with the training data, 
but does not perform well with the test dataset, then it is estimated that the model may be overfitted.

### Model Training and Evaluation
In this section, we’ll be training and testing 5 models, namely Logistic Regression, Random Forest, Decision Tree, K-Nearest Neighbors,
and XGBoost. We’ll also evaluate their performance at predicting loan defaults and their probability.

## Conclusion
In the classification report, accurancy is not a good measure of performance, because this is only accurate if the model is balanced. 
It will give inaccurate results if there is a class imbalance.

After oversampling data and training model, the best results among all models for this credit risk prediction is:

XGBoost (Bad loan recall =  63%, Good loan recall = 70%, Bad loan Precision = 21%, Good loan Precision = 93%) <br>that concludes:<br>  
    - Out of the actual bad loans, the model correctly predicted this for 63%.<br>
    - Out of the actual good loans the model correctly predicted this for 70%.<br>
    - Out of the good loans that the model predicted, 93% actually did.
    






