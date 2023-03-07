#!/usr/bin/env python
# coding: utf-8

# # KSHEMA S
# 
# TCS iON  INTERNSHIP
# RIO-125:HR Salary Dashboard - Train the Dataset and Predict Salary

# # Problem statement
# This project aims to sanitize the data, analysis and predict if an employee's salary is higher or lower than $50K/year depends on certain attributes using different ML classification algorithms.

# # Importing necessary libraries and dataset to the Python environment

# In[1]:


# Working with data
import numpy as np
import pandas as pd

# For Visualizations
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Loading the HR dataset 


# In[3]:


ds=pd.read_csv(r"C:\Users\Anish\Downloads\salarydata.csv")


# In[4]:


ds


# The dataset is shown here

# In[5]:


ds.describe()


# Dataset description
# 
# Age: Age of person
# 
# Workclass: Belongs to which working class like Private/government/self employed etc
# 
# Education: Person's maximum qualification
# 
# Education-Number: Numbered qualification
# 
# Salary: Traget coloumn
# 
#     

# In[6]:


# Shape of the dataset

print(ds.shape)


# # DATA cleaning
# 

# In[7]:


# Checking for null values in each coloumn


# In[8]:


print(ds.isna().sum())


# There is no null value in any of the coloumns

# In[9]:


# Check the datatypes of the data

ds.info()


# In[10]:


ds.nunique()


# In[11]:


ds['age'].unique() 


# In[12]:


ds['workclass'].unique() 


# In[13]:


ds['workclass'] = ds['workclass'].replace('?', np.nan)


# In[14]:


ds['workclass'].unique() 


# In[15]:


ds.apply(lambda col: col.unique())


# In[16]:


for col in ds:
    print(f'{col}: {ds[col].unique()}')


# The unique values in each coloumn have been displayed

# In[17]:


ds['occupation'].unique() 


# In[18]:


ds['occupation'] = ds['occupation'].replace('?', np.nan)
ds['native-country'] = ds['native-country'].replace('?', np.nan)


# In[19]:


print(ds.isna().sum())


# It is clear that workclass,occupation and native country contains null values

# In[20]:


ds['workclass'] = ds['workclass'].fillna(ds['workclass'].mode()[0])
ds['occupation'] = ds['occupation'].fillna(ds['occupation'].mode()[0])
ds['native-country'] = ds['native-country'].fillna(ds['native-country'].mode()[0])


# In[21]:


print(ds.isna().sum())


# The null values are replaced with mode of the data

# # Exploratory Data Analysis
# Univariate Analysis

# In[22]:


freqgraph = ds.select_dtypes(include = ['int'])
freqgraph.hist(figsize =(20,15))
plt.show()


# In[23]:


ds['relationship'].value_counts().plot.pie(autopct='%.0f%%')
plt.title("relationship")
plt.show()


# The employees with relationship shown majority are husbands followed by not in a family and own child

# In[24]:


sns.countplot(x= ds['salary'], palette="dark") 
#different types of credit accounts of a customer, shows the ability to handle multiple credits
plt.title("Salary scale")
plt.figure(figsize=(5,5))
plt.show()


# People are more who getting a salary of <=50K

# In[25]:


sns.countplot(x= ds['education'], palette="dark") 
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
#different types of credit accounts of a customer, shows the ability to handle multiple credits
plt.title("Education Qualification")
plt.figure(figsize=(10,10))

plt.show()


# More people have eductaional qualification as HS grad

# # Bivariate analysis (w.r.t. target coloumn salary)

# In[26]:


# Annual_Income vs credit score
sns.barplot(x=ds['age'], y=ds['salary'])
plt.title('Age vs Salary')
plt.show()


# In[27]:


sns.boxplot(y=ds['salary'], x=ds['education-num'])
plt.title('education-num vs salary')
plt.show()


# In[28]:


sns.catplot(x= 'sex', col= 'salary', data = ds, kind = 'count', col_wrap = 3)
plt.show()


# # Outlier detection and removal  using boxplot 

# In[29]:


num_col = ds.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(20,30))

for i, variable in enumerate(num_col):
                     plt.subplot(5,4,i+1)
                     plt.boxplot(ds[variable],whis=1.5)
                     plt.tight_layout()
                     plt.title(variable)


# In[30]:


# Identify the outliers and remove 

for i in num_col:
    Q1=ds[i].quantile(0.25) # 25th quantile
    Q3=ds[i].quantile(0.75) # 75th quantile
    IQR = Q3-Q1
    Lower_Whisker = Q1 - 1.5*IQR 
    Upper_Whisker = Q3 + 1.5*IQR
    ds[i] = np.clip(ds[i], Lower_Whisker, Upper_Whisker)


# In[31]:


# PLot the numerical columns
plt.figure(figsize=(20,30))
for i, variable in enumerate(num_col):
                     plt.subplot(5,4,i+1)
                     plt.boxplot(ds[variable],whis=1.5)
                     plt.tight_layout()
                     plt.title(variable)


# In[32]:


ds[['age','salary']].head(24)


# # Label Encoding

# In[33]:


from sklearn import preprocessing 
label= preprocessing.LabelEncoder()  
ds['workclass']=label.fit_transform(ds['workclass'])
ds['education']=label.fit_transform(ds['education'])
ds['occupation']=label.fit_transform(ds['occupation'])
ds['sex']=label.fit_transform(ds['sex'])



ds['race']=label.fit_transform(ds['race'])
ds['native-country']=label.fit_transform(ds['native-country'])
ds['marital-status']=label.fit_transform(ds['marital-status'])
ds['relationship']=label.fit_transform(ds['relationship'])


# In[34]:


ds


# In[35]:


for i in ['workclass', 'education','marital-status','occupation']:  
    ds[i]=label.fit_transform(ds[i])
    le_name_mapping =dict((zip(label.classes_, label.transform(label.classes_))))
    print(le_name_mapping)


# # Standardization

# In[36]:


scale_col = ['age',  'education-num', 'capital-gain',
       'capital-loss', 'hours-per-week']

from sklearn.preprocessing import StandardScaler

std = StandardScaler()

ds[scale_col]= std.fit_transform(ds[scale_col])


# In[37]:


ds


# In[38]:


ds.describe()


# In[39]:


ds.drop(['capital-gain','capital-loss','education-num'], axis = 1,inplace = True)
ds.head()


# Feature engineering
# 
# 
#   While analyzing the dataset,
# As we can see in 'descriptive statistics - Numerical columns',
# 'capital-gain'and 'capital-loss'  columns have 75% data as 0.00
#   - So, we can drop 'capital-gain'& 'capital-loss'  columns. 
# The column,education-num is the numerical version of the column education, so we also drop it.

# # Split dataset into test and train

# In[40]:



from sklearn.model_selection import train_test_split


# In[41]:


X = ds.drop('salary', axis=1)
y= ds['salary']


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.25, random_state=42, stratify=y)


# In[43]:


ds['salary'].value_counts()


# In[44]:


ds['marital-status'].value_counts()


# # Modelling
# 

# In[45]:


# split data into test and train
from sklearn.model_selection import train_test_split


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.25, random_state=42, stratify=y)


# In[47]:


print("Length of y train",len(y_train))
print("Length of y test",len(y_test))


# # 1) Logistic Regression
# In logistic regression, the model predicts the probability that an instance belongs to a particular class. This probability is represented by a value between 0 and 1, where 0 indicates that the instance definitely does not belong to the class and 1 indicates that it definitely does.To make these predictions, logistic regression uses a logistic function, which takes in a linear combination of the input features and maps it to a value between 0 and 1.

# In[48]:



from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,precision_score,recall_score,classification_report


# In[49]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=2000)
lr.fit(X_train,y_train)
pred_lr=lr.predict(X_test)
con_lr=confusion_matrix(y_test,pred_lr)
print("The confusion matrix of logistic regression is \n",con_lr)
ac_lr=accuracy_score(y_test,pred_lr)
print('Accuracy:',ac_lr*100)


# In[50]:


print(classification_report(y_test,pred_lr))


# *Precision is the fraction of predicted positive instances that are actually positive, and is calculated as TP / (TP + FP). It gives you an idea of the proportion of positive predictions that are correct. High precision means that the model is good at not labeling negative instances as positive.
# 
# *Recall is the fraction of actual positive instances that were predicted to be positive, and is calculated as TP / (TP + FN). It gives you an idea of the proportion of positive instances that the model was able to identify. High recall means that the model is good at finding all the positive instances.
# 
# *The F1 score is the harmonic mean of precision and recall, and is calculated as 2 * (precision * recall) / (precision + recall). It is a balanced metric that takes into account both precision and recall.
# Support is the number of instances in each class.
# 
# *Accuracy is the fraction of correct predictions made by the model, and is calculated as (TP + TN) / (TP + TN + FP + FN). It gives you an idea of the overall accuracy of the model.

# In[51]:


y_test


# In[52]:


pred_lr[:100]


# # 2) K Nearest Negihbour Classifier

# In[53]:


from sklearn.neighbors import KNeighborsClassifier
acc_values=[]
neighbors=np.arange(70,90)
for k in neighbors:
    knn=KNeighborsClassifier(n_neighbors=k, metric='minkowski')
    knn.fit(X_train, y_train)
    pred_knn=knn.predict(X_test)
    acc=accuracy_score(y_test, pred_knn)
    acc_values.append(acc)
    
     


# In[54]:


plt.plot(neighbors,acc_values,'o-')
plt.xlabel('k value')
plt.ylabel('accuracy')
     


# In[55]:



print(classification_report(y_test, pred_knn))


# In[56]:


pred_knn[:20]


# In[57]:


con_lr=confusion_matrix(y_test,pred_knn)
print("The confusion matrix of knn is \n",con_lr)
ac_knn=accuracy_score(y_test,pred_knn)
print('Accuracy:',ac_knn*100)


# # 3)Decision Tree classifier

# In[58]:


from sklearn.tree import DecisionTreeClassifier
dtr=DecisionTreeClassifier()
dtr.fit(X_train,y_train)
dtr.fit(X_train,y_train)
pred_dt=dtr.predict(X_test)
con_dtr=confusion_matrix(y_test,pred_dt)
print("The confusion matrix of decision tree is \n",con_dtr)
ac_dt=accuracy_score(y_test,pred_dt)
print('Accuracy:',ac_dt*100)


# In[59]:


print(classification_report(y_test, pred_dt))


# # 4)Support Vector Machine

# In[60]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
pred_svc=svc.predict(X_test)
con_svc=confusion_matrix(y_test,pred_svc)
print("The confusion matrix of decision tree is \n",con_svc)
ac_svc=accuracy_score(y_test,pred_svc)
print('Accuracy:',ac_svc*100)


# In[61]:


print(classification_report(y_test, pred_svc))


# In[62]:


pred_svc[:50]


# # 5)Random Forest Classifier

# In[63]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
pred_RFC=rf.predict(X_test)
con_rf=confusion_matrix(y_test,pred_RFC)
print("The confusion matrix of random forest is \n",con_rf)
ac_rf=accuracy_score(y_test,pred_RFC)
print('Accuracy:',ac_rf*100)


# In[64]:


print(classification_report(y_test, pred_RFC))
     


# # 6) GradientBoostingClassifier
# 
# 

# In[65]:


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(X_train,y_train)
pred_gb = gb.predict(X_test)

print('Classification_report is')
print(classification_report(y_test,pred_gb))


# In[66]:




# # 7) Naive_bayes Classifier

# In[67]:





# In[68]:




# #    Comparisaon of accuracies of different models 

# In[69]:



# In[70]:



# In[71]:


#
# Gradient Booster  gives best accuracy compared to other supervised learning algorithms.
# For salary prediction,gradient booster is selected.

# In[72]:


ds


# In[73]:


# save the model
import pickle
filename = 'model.pkl'
pickle.dump(gb, open(filename, 'wb'))


# In[74]:


load_model = pickle.load(open(filename,'rb'))


# In[75]:


load_model.predict([[.03,4,11,4,3,5,4,0,0.1,34]])


# In[76]:


load_model.predict([[33,4,11,4,3,0,4,1,30,34]])


# In[77]:


load_model.predict([[.99,11,4,2,3,5,4,0,-0.19,38]])


# In[78]:


load_model.predict([[50,3,11,6,4,4,4,0,32,9]])


# In[ ]:





# In[ ]:




