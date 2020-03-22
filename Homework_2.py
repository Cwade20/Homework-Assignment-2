
# coding: utf-8

# In[70]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[71]:


xls = pd.ExcelFile('UniversalBank.xlsx')
df = pd.read_excel(xls, 'Data')


# In[72]:


df1=df[df['Personal Loan']==0]
df2=df[df['Personal Loan']==1]
plt.scatter('Age','Income',label='Loan not accepted',data=df1)
plt.scatter('Age','Income',label='Loan accepted',data=df2)
plt.xlabel('Age in years')
plt.ylabel('Income in $000')
plt.title('Loan Acceptance classification based on age and income')
plt.legend()
plt.show()


# In[73]:


x=df.drop(['ID'],axis =1)
x=x.drop(['ZIP Code'],axis =1)
x=x.drop(['Personal Loan'],axis =1)
y=df['Personal Loan']


# In[74]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)


# Part 2 (a):-

# In[75]:


model = LogisticRegression()
mfit=model.fit(x_train, y_train)
y_pred=mfit.predict(x_test)
y_pred


# In[76]:


print(confusion_matrix(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(np.array(y_test), np.array(y_pred), labels=[0,1]).ravel()
print("Accuracy of class 1",tp/(tp+fn))
print("Accuracy of class 0",tn/(fp+tn))
score=accuracy_score(y_test, y_pred)
print("Accuracy score",score)


# Part 2 (c):-

# In[77]:


random_subset = df.sample(n=100)
random_subset
random_subset=pd.DataFrame(random_subset)
len(random_subset[random_subset['Personal Loan']==1])


# Part 2 (d):-

# In[84]:



y_probas=mfit.predict_proba(x_test)[:,1]
label= [0 if y_score<0.5 else 1 for y_score in y_probas]
data1=pd.DataFrame(y_probas,columns=['Probability'])
data1['label']=label
data1=data1.sort_values(by=['Probability'],ascending=False)
data1=data1.reset_index()
data1=data1.drop(['index'],axis=1)

dec=int((100*10)/100)
lift_data=pd.DataFrame()
counter=1
decile=[]
no_of_cases=[]
label1_count=0

for i in range(0,100,dec):
    decile.append(counter)
    counter=counter+1
    no_of_cases.append(dec)
lift_data['decile']=decile
lift_data['no_of_cases']=no_of_cases  
no_of_actualcases=[]
for i in range(0,100,10):
    x=len(data1[data1['label']==1].loc[i:i+9])
    no_of_actualcases.append(x)
lift_data['no_of_actualcases']=no_of_actualcases
cum_events=[]
for i in range(10,110,10):
    x=len(data1[data1['label']==1].loc[:i-1])
    cum_events.append(x)
lift_data['cum_events']=cum_events
gain=[]
for i in range(len(lift_data)):
    x=(lift_data['cum_events'][i]/100)*100
    gain.append(x)
lift_data['gain']=gain
lift=[]
count=10
for i in range(len(lift_data)): 
    x=lift_data['gain'][i]/(count)
    count=count+10
    lift.append(x)
lift_data['lift']=lift


# In[85]:


lift_data


# In[79]:


import scikitplot as skplt
y_probas=mfit.predict_proba(x_test)
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()


# Part 2 e:-

# In[87]:


print("The percentage of customers who were incorrectly classified",(fn/(fn+tp)))


# Part 3:-

# In[92]:


decisions = (model.predict_proba(x_test)[:,1] >= 0.3).astype(int)


# In[93]:


tn, fp, fn, tp = confusion_matrix(np.array(y_test), decisions, labels=[0,1]).ravel()


# In[94]:


print("Accuracy of class 1",tp/(tp+fn))
print("Accuracy of class 0",tn/(fp+tn))
score=accuracy_score(y_test, y_pred)
print("Accuracy score",score)

