#!/usr/bin/env python
# coding: utf-8

# # Attrition Dataset

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tabulate as tb


# In[78]:


att = pd.read_csv(r'G:\Data Scientist_ Recruitment_April 2023\Employee Attrition.csv')


# In[79]:


att.head()   #Checking how data looks


# In[80]:


att.isnull().sum()[att.isnull().sum()>0]        #Checking Null Values


# In[81]:


att.columns[att.dtypes == 'object']


# In[82]:


att.shape


# In[83]:


att.Over18.value_counts()


# In[84]:


att.EmployeeCount.value_counts()


# In[85]:


att.StandardHours.value_counts()


# # Data Cleaning 

# In[86]:


att = att.drop(['Over18','EmployeeCount','StandardHours','EmployeeNumber'],axis =1)      #same value in all records


# In[87]:


att.Attrition = att.Attrition.replace({'Yes':1,'No':0})


# In[88]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[89]:


att[att.select_dtypes(include = 'object').columns] = att[att.select_dtypes(include = 'object').columns].apply(le.fit_transform)


# In[90]:


att.columns[att.dtypes == 'object']


# # Sampling

# In[91]:


from sklearn.model_selection import train_test_split


# In[92]:


att_train,att_test = train_test_split(att,test_size=.3)   #split it into train and test in 70:30 ratio


# In[93]:


att_train_x = att_train.drop(['Attrition'],axis =1)
att_train_y = att_train['Attrition']
att_test_x = att_test.drop(['Attrition'],axis =1)
att_test_y = att_test['Attrition']


# In[94]:


print(att_train_x.shape,att_train_y.shape,att_test_x.shape)


# # Model Building : Decision Tree

# In[95]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(class_weight='balanced', criterion='entropy',max_depth=3)


# In[96]:


dt.fit(att_train_x,att_train_y)


# In[97]:


pred = dt.predict(att_test_x)


# In[98]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score


# In[99]:


tab = confusion_matrix(att_test_y,pred)
tab


# In[100]:


# array([[219,  27],
#       [ 32,  16]], dtype=int64)    here we are facing problem of class imbalance 
# as records for class 0 are more than class 1 , so model giving wrong predictions for class 1 
# to overcome this problem we will do oversampling using "class_wieght = 'balanced'" and re-run the model


# In[101]:


accuracy_score(att_test_y,pred)*100


# In[102]:


precision_score(att_test_y,pred)*100


# In[103]:


recall_score(att_test_y,pred)*100


# In[104]:


# pricision and recall is low , tune hyperparameter accordingly to increase them


# In[105]:


dt.feature_importances_


# In[106]:


df = pd.DataFrame()
df['features'] = att_train_x.columns
df['score']= dt.feature_importances_*100
df.sort_values('score',ascending=False)


# In[ ]:





# # Model Building 

# In[107]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight='balanced', criterion='entropy',max_depth=7)


# In[108]:


rf.fit(att_train_x,att_train_y)


# In[109]:


pred1 = rf.predict(att_test_x)


# In[110]:


tab1 = confusion_matrix(att_test_y,pred1)
tab1


# In[121]:


accuracy_score(att_test_y,pred1)*100


# In[112]:


precision_score(att_test_y,pred1)


# In[113]:


recall_score(att_test_y,pred1)


# In[114]:


rf.feature_importances_


# In[115]:


df1 = pd.DataFrame()
df1['features'] = att_train_x.columns
df1['score']= rf.feature_importances_*100
df1.sort_values('score',ascending=False)


# In[ ]:





# In[116]:


data = att[['MonthlyIncome','Age','OverTime','DailyRate','YearsWithCurrManager','JobRole','MonthlyIncome','StockOptionLevel','Attrition']]


# In[117]:


data.head()


# In[118]:


data_train,data_test = train_test_split(data,test_size=.2)


# In[119]:


data_train_x = data_train.drop(['Attrition'],axis =1)
data_train_y = data_train['Attrition']
data_test_x = data_test.drop(['Attrition'],axis =1)
data_test_y = data_test['Attrition']


# In[120]:


dt.fit(data_train_x,data_train_y)


# In[48]:


pred2 = dt.predict(data_test_x)


# In[49]:


tab2 = confusion_matrix(data_test_y,pred2)
tab2


# In[50]:


accuracy_score(data_test_y,pred2)


# In[51]:


precision_score(data_test_y,pred2)


# In[52]:


recall_score(data_test_y,pred2)


# In[53]:


#### did feature selection using Decision tree and builded a model


# In[54]:


#1)Identify the Key factors influencing attrition behavior
data.columns


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[123]:


df = pd.read_csv(r'G:\Data Scientist_ Recruitment_April 2023\Employee Attrition.csv')
df.head()


# In[124]:


df.groupby('Attrition')['Attrition'].agg('count').plot.bar()


# In[125]:


df.Attrition_numeric = df.Attrition
df.loc[df.Attrition == 'Yes','Attrition_numeric'] = 1
df.loc[df.Attrition == 'No','Attrition_numeric'] = 0

plt.figure(figsize=(12,8))
sns.barplot(x = 'Gender', y = 'Attrition_numeric', data=df)


# In[126]:


plt.figure(figsize=(12,8))
sns.barplot(x = 'EducationField', y = 'Attrition_numeric', data=df)


# In[127]:


plt.figure(figsize=(12,8))
sns.barplot(x = 'OverTime', y = 'Attrition_numeric', data=df)


# In[128]:


plt.figure(figsize=(12,8))
sns.barplot(x = 'JobInvolvement', y = 'Attrition_numeric', data=df)


# In[ ]:





# In[ ]:





# In[ ]:




