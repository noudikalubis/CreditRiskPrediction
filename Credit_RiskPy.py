#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Prediction

# In this project, we will create machine learning models that can predict the possibility of a loss happening due to a borrower's failure to repay a loan or to satisfy contractual obligations.

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Load data
loan_data = pd.read_csv('loan_data_2007_2014.csv')


# In[3]:


# Investigating Data Set
loan_data.head()


# In[4]:


loan_data.tail()


# In[5]:


# Check the number of rows
len(loan_data)


# ### Define Target Variable

# Target Variable is the feature of a dataset about which we want to gain a deeper understanding and learn patterns between the dataset.

# In[6]:


# Define target variable
# In this dataset 'loan_status' is the target variable
loan_data.loan_status.value_counts()


# Because the goal is to predict whether the loans is risky or not, we need to see data of each loans hystorically. 
# In this dataset, we can see that in loan_status columns. Then, we can classify them as follows:
# 
# (This classification depends on the regulations of the lending company)
#     
#     • Good loans : 'Current', 'Fully Paid', ''Does not meet the credit policy. Status:Fully Paid'
#     • Bad loans  : 'Charged Off', 'Late (31-120 days)',  'In Grace Period', 'Late (16-30 days)', 'Default', 'Does not meet
#                     the credit policy. Status:Charged Off'

# In[7]:


# Define values, classify good loans based on the dataset
good_loans = ['Current', 'Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']


# In[8]:


# Classify the types of loans
loan_data['loan_types'] = np.where(loan_data['loan_status'].isin(good_loans), 1, 0)


# In[9]:


# Visualize the comparison
plt.title('Good vs Bad Loans')
sns.barplot(x=loan_data.loan_types.value_counts().index,y=loan_data.loan_types.value_counts().values)


# ### Feature engineering & Feature Selection 

# What we need to do:
#     
#     • Checking the dataset for missing values, duplicate, inconsistent columns and etc.
#         
#         After that, we can do this following:
#         
#         • Drop identities columns (which have each unique values but irrelevant)
#         • Drop columns that contain >50% missing values.
#         • Drop columns that only have 1 unique values.
#         • Drop columns that contain data leakage.
#         • Drop columns that have similar values (Check the correlation).

# In[10]:


# Check information about the DataFrame.
loan_data.info()


# In[11]:


# Drop irrelevant columns
drop_identities = ['Unnamed: 0', 'id', 'member_id', 'url', 'desc', 'sub_grade', 'emp_title', 'url', 'title']
loan_data.drop(columns=drop_identities, axis=1, inplace=True)


# In[12]:


# Get columns that have more than 50% missing values
na_values = loan_data.isnull().mean()
na_values[na_values>0.5]


# In[13]:


# Filtering data with less than 2 unique values
loan_data.nunique()[loan_data.nunique() < 2].sort_values()


# In[14]:


# Drop the irrelevant columns
drop_col = ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog', 'annual_inc_joint', 'dti_joint'
            , 'verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
           'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl',
           'inq_last_12m', 'max_bal_bc', 'application_type', 'policy_code']
loan_data.drop(columns=drop_col, axis=1, inplace=True)


# ### Data Leakage 

# One of the common pitfalls when building a machine learning model is Data Leakage. Usually the target leakage will have much higher feature importance. 
# 
# In this dataset the example of target leakage is 'out_prncp', when'out_prncp' column has 0 value, it means the loan is already fully paid. The result of this column will be super accurate because it's already happened, the loan is already paid, but this data is not relevant to predict a new borrower so it will be unfair. This kind of variable will create overly optimistic models that are practically useless and cannot be used in production.
# 
# Hence, the columns that contain Data Leakage will be drop.

# In[15]:


# Drop columns that contain data leakage
leakage_col = ['issue_d', 'loan_status', 'pymnt_plan', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 
                'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
                'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d']

loan_data.drop(columns=leakage_col, axis=1, inplace=True)


# In[16]:


#Check correlation
plt.figure(figsize=(24,24))
sns.heatmap(loan_data.corr(), annot=True, annot_kws={'size':14})


# From this heatmap, we can see the columns that are similar with each other. 

# In[17]:


# Checking similar columns
loan_data[['loan_amnt','funded_amnt','funded_amnt_inv']].describe()


# In[18]:


# Drop 2 of the similar columns
loan_data.drop(columns = ['funded_amnt', 'funded_amnt_inv'], inplace = True)


# Check the Missing Values again.

# In[19]:


#Checking for missing values
loan_data.isnull().sum()


# Check the collumns that have the same number of missing values.

# In[20]:


# Check tot_coll_amt, tot_cur_bal, total_rev_hi_lim
tot_cols = ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']
loan_data[tot_cols].sample(10)


# In[21]:


loan_data[tot_cols].describe()


# In[22]:


# Visualize data distribution of each columns
loan_data.boxplot(column=['tot_coll_amt'])
plt.show()


# In[23]:


loan_data.boxplot(column=['tot_cur_bal'])
plt.show()


# In[24]:


loan_data.boxplot(column=['total_rev_hi_lim'])
plt.show()


# Conclusion:
# - The description of these columns are rather vague
# - Total missing value is 70276
# - The rows that contain missing values in those columns will be dropped

# In[25]:


# Drop all rows that contain missing value from tot_coll_amt, tot_cur_bal, total_rev_hi_lim columns
loan_data.dropna(subset = ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim'], inplace = True)


# In[26]:


# Reset index
loan_data.reset_index(drop= True, inplace = True)


# Check the data with less than 10 unique values.

# In[27]:


# Filtering data with less than 10 unique values
loan_data.nunique()[loan_data.nunique() < 10].sort_values()


# In[28]:


def risk_pct(x):
    ratio = (loan_data.groupby(x)['loan_types'] # group by
         .value_counts(normalize=True) # calculate the ratio
         .mul(100) # multiply by 100 to be percent
         .rename('risky (%)') # rename column as percent
         .reset_index())

    sns.lineplot(data=ratio[ratio['loan_types'] == 0], x=x, y='risky (%)')
    plt.title(x)
    plt.show()


# In[29]:


print(loan_data.nunique()[loan_data.nunique() < 12].sort_values().index)


# In[30]:


#unique columns and months date column
unique_cols = ['term', 'initial_list_status', 'verification_status',
       'home_ownership', 'acc_now_delinq', 'grade', 'inq_last_6mths',
       'collections_12_mths_ex_med', 'emp_length']
for cols in unique_cols:
    risk_pct(cols)


# In[31]:


# Check columns with numerical data type
num_dtype = loan_data.select_dtypes(exclude= 'object')
num_dtype.columns


# In[32]:


# Check columns with string data type
num_dtype = loan_data.select_dtypes(include= 'object')
num_dtype.columns


# The following columns should be modified because they don't have appropriate data types.

# In[33]:


mod_cols = ['term', 'emp_length', 'earliest_cr_line', 'last_credit_pull_d']
loan_data[mod_cols]


# In[34]:


# Convert 'term' column to numerical datatype and replace months with empty string
loan_data['term'] = pd.to_numeric(loan_data['term'].str.replace(' months', ''))
loan_data['term']


# In[35]:


# Check unique values of 'emp_length' column
loan_data['emp_length'].unique()


# In[36]:


# Change 'emp_map' to int data type
emp_map = {
    '< 1 year' : '0',
    '1 year' : '1',
    '2 years' : '2',
    '3 years' : '3',
    '4 years' : '4',
    '5 years' : '5',
    '6 years' : '6',
    '7 years' : '7',
    '8 years' : '8',
    '9 years' : '9',
    '10+ years' : '10'
}

loan_data['emp_length'] = loan_data['emp_length'].map(emp_map).fillna('0').astype(int)
loan_data['emp_length'].unique()


# In[37]:


# Displays 'earliest_cr_line' column
loan_data['earliest_cr_line']


# In[38]:


# Convert to date data type & assign them into new column
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format = '%b-%y')


# In[39]:


# Assuming we are in January 2016, Calculate the months between 2 dates and change them to num data type
loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2016-01-30') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')))


# In[40]:


loan_data['mths_since_earliest_cr_line'].describe()


# In[41]:


# Check variable that have (-) value
loan_data.loc[: , ['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']][loan_data['mths_since_earliest_cr_line'] < 0]


# In[42]:


# Replace the year 20.. to 19..
loan_data['earliest_cr_line_date'] = loan_data['earliest_cr_line_date'].astype(str)
loan_data['earliest_cr_line_date'][loan_data['mths_since_earliest_cr_line'] < 0] = loan_data['earliest_cr_line_date'][loan_data['mths_since_earliest_cr_line'] < 0].str.replace('20','19')


# In[43]:


loan_data['earliest_cr_line_date'][843]


# In[44]:


# Change to date data type
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line_date'])
loan_data['earliest_cr_line_date']


# In[45]:


loan_data['mths_since_earliest_cr_line']


# In[46]:


# Check the data again (Assuming we're in Jan 2016)
loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2016-01-30') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')))
loan_data['mths_since_earliest_cr_line'].describe()


# In[47]:


loan_data.drop(columns = ['earliest_cr_line_date' , 'earliest_cr_line'], inplace = True)


# In[48]:


loan_data['last_credit_pull_d']


# In[49]:


# Change to date data type and fill NaN data with max date
loan_data['last_credit_pull_d'] = pd.to_datetime(loan_data['last_credit_pull_d'], format = '%b-%y').fillna(pd.to_datetime("2016-01-30"))


# In[50]:


# Calculate the months between 2 dates and change them to num data type
loan_data['mths_since_last_credit_pull_d'] = round(pd.to_numeric((pd.to_datetime('2016-01-30') - loan_data['last_credit_pull_d']) / np.timedelta64(1, 'M')))


# In[51]:


loan_data['mths_since_last_credit_pull_d'].describe()


# In[52]:


# Drop column last_credit_pull_d 
loan_data.drop(columns = ['last_credit_pull_d'], inplace = True)


# After the columns have fitting data types, we can check the missing values again.

# In[53]:


#Checking for missing values
loan_data.isnull().sum()


# In[54]:


loan_data.drop(columns ='zip_code', inplace = True)


# In[55]:


# Drop rows that contain missing value
loan_data.dropna(subset = ['revol_util'], inplace = True)
loan_data.reset_index(drop= True, inplace = True)


# In[56]:


#Checking for missing values
missing_value = loan_data.isnull().sum()
missing_value[missing_value>0]


# In[57]:


loan_data.info()


# In[58]:


#Check correlation
plt.figure(figsize=(24,24))
sns.heatmap(loan_data.corr(), annot=True, annot_kws={'size':14})


# ### One Hot Encoding

# One hot encoding is a technique that we use to represent categorical variables as numerical values in a machine learning model.
# This technique can improve model performance by providing more information to the model about the categorical variable.

# In[59]:


# Convert categorical columns with One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
cat_cols = [col for col in loan_data.select_dtypes(include='object').columns.tolist()]
onehot_cols = pd.get_dummies(loan_data[cat_cols], drop_first=True)


# In[60]:


onehot_cols


# ### Standardization

# Standardization entails scaling data to fit a standard normal distribution.
# 
# A standard normal distribution is defined as a distribution with a mean of 0 and a standard deviation of 1

# In[61]:


# All numeric columns are standardized with StandardScaler
from sklearn.preprocessing import StandardScaler

num_cols = [col for col in loan_data.columns.tolist() if col not in cat_cols + ['loan_types']]
ss = StandardScaler()
std_cols = pd.DataFrame(ss.fit_transform(loan_data[num_cols]), columns=num_cols)


# In[62]:


std_cols


# ### Final Data

# Get the final data by combining it all.

# In[63]:


# Get final data
final_data = pd.concat([onehot_cols, std_cols, loan_data[['loan_types']]], axis=1)
final_data.head()


# In[64]:


final_data.loan_types.value_counts()


# In[65]:


# Checking class imbalance with final data
plt.title('Good vs Bad Loans')
sns.barplot(x=final_data.loan_types.value_counts().index,y=final_data.loan_types.value_counts().values)


# ### Data Splitting

# We can split the data to easily evaluate the performance of our model. Such as, if it performs well with the training data, but does not perform well with the test dataset, then it is estimated that the model may be overfitted.
# 
# Split data into train and test, with a comparison of 80% for training data and 20% for testing data. 

# In[66]:


# Data Splitting
X = final_data.drop('loan_types', axis = 1)
y = final_data['loan_types']


# In[67]:


# Spliting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[68]:


X_train.shape, X_test.shape


# In[69]:


# Checking  imbalance for training dataset
y_train.value_counts()


# ### Oversampling Minority Class to Resolve Class Imbalance

# We can see above that our data is imbalanced, so we can do oversampling for the minority class.

# In[70]:


# Oversampling Minority Class
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

#check value counts before and after oversampling
print('Before OverSampling:\n{}'.format(y_train.value_counts()))
print('\nAfter OverSampling:\n{}'.format(y_train_ros.value_counts()))


# ### Train the Model

# In[71]:


# Import Library
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# 1. Logistic Regression

# In[72]:


# Logistic Regression
# Building Model
LR_ros= LogisticRegression(max_iter=600)  
LR_ros.fit(X_train_ros, y_train_ros)

# Predicting
y_pred_LR_ros = LR_ros.predict(X_test)

# Classification report
target_names = ['bad loan', 'good loan']
print('Classification_Report:')
print(classification_report(y_test, y_pred_LR_ros, digits=4, target_names = target_names))


# 2. Random Forest

# In[73]:


# Random Forest
# Building model
rf_ros = RandomForestClassifier(max_depth=10, n_estimators=20)
rf_ros.fit(X_train_ros, y_train_ros)

# Predicting
y_pred_rf_ros = rf_ros.predict(X_test)

# Classification report
target_names = ['bad loan', 'good loan']
print('Classification_Report:')
print(classification_report(y_test, y_pred_rf_ros, digits=4, target_names = target_names))


# 3. Decision Tree

# In[74]:


# Decision Tree
# Building model
dt_ros = DecisionTreeClassifier(max_depth = 10)
dt_ros.fit(X_train_ros, y_train_ros)

# Predicting
y_pred_dt_ros = dt_ros.predict(X_test)

# Classification report
target_names = ['bad loan', 'good loan']
print('Classification_Report:')
print(classification_report(y_test, y_pred_dt_ros, digits=4, target_names = target_names))


# 4. K-Nearest Neighbors

# In[75]:


# K-Nearest Neighbors
# Building model
knn_ros = KNeighborsClassifier(n_neighbors=20)
knn_ros.fit(X_train_ros, y_train_ros)

# Predicting
y_pred_knn_ros = knn_ros.predict(X_test)

# Classification report
target_names = ['bad loan', 'good loan']
print('Classification_Report:')
print(classification_report(y_test, y_pred_knn_ros, digits=4, target_names = target_names))


# 5. XGBoost

# In[76]:


# XGBoost
# Building model
from xgboost import XGBClassifier
xgb_ros = XGBClassifier(max_depth=5)
xgb_ros.fit(X_train_ros, y_train_ros)

# Predicting
y_pred_xgb_ros = xgb_ros.predict(X_test)

# Classification report
target_names = ['bad loan', 'good loan']
print('Classification_Report:')
print(classification_report(y_test, y_pred_xgb_ros, digits=4, target_names = target_names))


# ## Conclusion

# • In the classification report, accurancy is not a good measure of performance, because this is only accurate if the model is balanced. It will give inaccurate results if there is a class imbalance.
# 
# • After oversampling data and training model, the best results among all models for this credit risk prediction is:
# 
#     1. XGBoost (Bad loan recall =  63%, Good loan recall = 70%, Bad loan Precision = 21%, Good loan Precision = 93%) 
#        that concludes:  
#         - Out of the actual bad loans, the model correctly predicted this for 63%.
#         - Out of the actual good loans the model correctly predicted this for 70%.
#         - Out of the good loans that the model predicted, 93% actually did.
#         
#     2. Random Forest (Bad loan recall =  64%, Good loan recall = 67%, Bad loan Precision = 19%, Good loan Precision = 93%)
#        that concludes:
#         - Out of the actual bad loans, the model correctly predicted this for 64%.
#         - Out of the actual good loans the model correctly predicted this for 67%.
#         - Out of the good loans that the model predicted, 93% actually did.
