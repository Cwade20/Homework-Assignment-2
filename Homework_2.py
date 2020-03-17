
# coding: utf-8

# In[97]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[31]:


xls = pd.ExcelFile('UniversalBank.xlsx')
df = pd.read_excel(xls, 'Data')


# In[69]:


len(df[df['Personal Loan']==1])


# In[20]:


df1=df[df['Personal Loan']==0]
df2=df[df['Personal Loan']==1]
plt.scatter('Age','Income',label='Loan not accepted',data=df1)
plt.scatter('Age','Income',label='Loan accepted',data=df2)
plt.xlabel('Age in years')
plt.ylabel('Income in $000')
plt.title('Loan Acceptance classification based on age and income')
plt.legend()
plt.show()


# In[35]:


x=df.drop(['ID'],axis =1)


# In[37]:


x=df.drop(['ZIP Code'],axis =1)


# In[38]:


y=df['Personal Loan']


# In[43]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)


# In[96]:


model = LogisticRegression()
mfit=model.fit(x_train, y_train)
y_pred=mfit.predict(x_test)


# In[79]:


x=0
for i in y_test:
    if i ==0:
        x=x+1
    else:
        pass
print(x)


# In[87]:


tn, fp, fn, tp = confusion_matrix(np.array(y_test), np.array(y_pred), labels=[0,1]).ravel()
#print("specificity ",1338/(1338+0))
#print("sensitivity ",160/(160+2))
print(tn)
print(fp)
print(fn)
print(tp)


# In[98]:


score=accuracy_score(y_test, y_pred)
print(score)


# In[54]:


random_subset = df.sample(n=100)
random_subset=pd.DataFrame(random_subset)


# In[58]:


len(random_subset[random_subset['Personal Loan']==1])


# In[59]:


df1=random_subset[random_subset['Personal Loan']==1]


# In[61]:


x1=df1.drop(['ID'],axis =1)
x1=df1.drop(['ZIP Code'],axis =1)
y1=df1['Personal Loan']


# In[65]:


y1_pred=mfit.predict(x1)


# In[66]:


print(confusion_matrix(y1, y1_pred))


# Part 3:-

# In[90]:


decisions = (model.predict_proba(x_test)[:,1] >= 0.3).astype(int)


# In[91]:


decisions


# In[93]:


tn, fp, fn, tp = confusion_matrix(np.array(y_test), decisions, labels=[0,1]).ravel()


# In[94]:


print(tn)
print(fp)
print(fn)
print(tp)


# In[99]:


score1=accuracy_score(y_test, decisions)
print(score1)

