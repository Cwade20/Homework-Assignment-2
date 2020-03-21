# Homework-Assignment-2 

Task - Case: Acceptance of Loan Offers
Universal Bank is a relatively young bank growing rapidly in terms of overall customer acquisition. The
majority of these customers are liability customers (depositors) with varying sizes of relationship with
the bank. The customer base of asset customers (borrowers) is quite small, and the bank is interested in
expanding this base rapidly to bring in more loan business. In particular, it wants to explore ways of
converting its liability customers to personal loan customers (while retaining them as depositors).

A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over
9%. This has encouraged the retail marketing department to devise smarter campaigns with better
target marketing. Your goal is to model the previous campaign’s customer behavior to analyze what
combination of factors make a customer more likely to accept a personal loan. This will serve as the
basis for the design of a new campaign.

The file UniversalBank.xls contains data on 5000 liability customers of Universal Bank who were targeted
in the previous personal loan campaign. The data include customer demographic information (age,
income etc.), the customer’s relationship with the bank (mortgage, securities account etc.), and the
customer response to the last campaign (Personal Loan). A 1 in the Personal Loan column indicates the
loan offer was accepted. The descriptions of the variables are in the Description worksheet in the file.
(Read the descriptions to get a better idea about the variables.) 
Use the dataset to answer the following questions. (Hint: In the dataset, since the row containing
attribute names is not the first row of the worksheet, inspect your dataset, set your dataframes correctly,
or adjust your data set.)

## 1. Create a scatterplot of Age vs. Income, using color to differentiate customers who accepted the loan and those who did not. Which variable (i.e., age or income) appears to be potentially more useful in classifying customers? Explain.

![image](https://user-images.githubusercontent.com/61456930/77213790-7e292400-6ae2-11ea-897a-e90270ed7976.png)

```python
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#Standard import functions for pythond coding

xls = pd.ExcelFile("/Users/Charles/Desktop/GMU/GBUS738/UniversalBank.xlsx")
df = pd.read_excel(xls, 'Data')
#Imports and reads our data files, also denotes file as df / xls for remainder of code

len(df[df['Personal Loan']==1])
df1=df[df['Personal Loan']==0]
df2=df[df['Personal Loan']==1]
#Denotes those who did NOT accept a personal loan as a 0, and those who did as a 1. Utilized for sorting later

plt.scatter('Age','Income',label='Loan not accepted',data=df1)
plt.scatter('Age','Income',label='Loan accepted',data=df2)
#Plots 2 independent charts on 1 graph. 1 chart for those who did not accept (0 as denoted in line 44), and those who did (1 in line 45)

plt.xlabel('Age in years')
plt.ylabel('Income in $000')
#adds X and Y axis labels

plt.title('Loan Acceptance classification based on age and income')
#adds plot title 

plt.legend()
#adds legend

plt.show()
#returns plot 
```

>Based on the information from UniversalBank, and our scatterplot, it does appear there is a stronger correlation between Income and loan acceptance, as compared to age and loan acceptance. Based on visually looking at our scatter plot, we can see that plot points listed in orange represent an individual taking a given loan, where blue represents those not taking the loan. While we can see there is a spread across all age groups of those taking or not taking loans, we can see a stronger correlation between income and those who accepted loans. It does appear that those individuals who have an income of >$100,000, are more strongly correlated with loan acceptance, as compared to those who have an income of <$100,000. Based on the graph, those who have an income sub $100,000, had very few loans, where as the majority of loans were above the income range of $100,000. 

## 2. Build a logistic regression model to classify customers into those who are likely to accept personal loan offer and those who are not. Use all the available variables as predictors except ID and ZIP Code. (Hint: Since the Logistic Regression operator expects binominal or polynominal target variables, if the target variable is numeric, you will have to convert it to binominal by using the Numerical to Binominal operator.)

 - a. Evaluate the overall predictive accuracy of the model as well as the accuracy of each class
using appropriate metrics.

```python

x=df.drop(['ID'],axis =1)
x=df.drop(['ZIP Code'],axis =1)

y=df['Personal Loan']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.7)
model = LogisticRegression()
mfit=model.fit(x_train, y_train)
y_pred=mfit.predict(x_test)
x=0
for i in y_test:
    if i ==0:
        x=x+1
    else:
        pass
print(x)
tn, fp, fn, tp = confusion_matrix(np.array(y_test), np.array(y_pred), labels=[0,1]).ravel()

print(tn)
print(fp)
print(fn)
print(tp)
score=accuracy_score(y_test, y_pred)
print(score)
```

> Confusion Matrix

>tn =3167
>fp =0
>fn =2
>tp =331


>Utilizing the results from our Confusion Matrix, we can generate an overall accuracy report, as well as accuracy of each class:

```python
accuracy=(3167+331)/(3167+0+2+331)
print(accuracy)

accuracy_class0=3167/(3167+0)
print(accuracy_class0)

accuracy_class1=331/(2+331)
print(accuracy_class1)
```

>Overall Accuracy = 0.9994285714285714

>Accuracy of predicted class 0 = 1.0

>Accuracy of predicted class 1 = 0.993993993993994


 - b. What was the default cutoff probability used to generate the classifications?
 >.5
 

-  c. Assuming that the dataset contains a representative sample of the liability customers of
the bank, if you target 100 customers randomly (i.e., without the aid of any predictive
model), how many of them would potentially accept a personal loan offer?
```python
random_subset = df.sample(n=100)
#calls for a random sample of 100

random_subset=pd.DataFrame(random_subset)
#puts name to random subset 
len(random_subset[random_subset['Personal Loan']==1])
df1=random_subset[random_subset['Personal Loan']==1]

x1=df1.drop(['ID'],axis =1)
x1=df1.drop(['ZIP Code'],axis =1)
#calls to drop ID and Zip code from the column

y1=df1['Personal Loan']
y1_pred=mfit.predict(x1)
print(confusion_matrix(y1, y1_pred))


```
>7 Total people would accept the loan offer


-  d. Now if you use your model in Part (2) to select 100 customers with the highest probability
of loan acceptance, how many of them would potentially accept a personal loan offer?
(Hint: Revise the process from Part (2) to generate a lift chart.)



 - e. What percentage of customers who accepted the loan were incorrectly classified by the
model in Part (2)?
>.1


## 3. Suppose the bank is interested in improving the accuracy of identifying the potential positive responders, i.e., those who would accept the loan offer. Create a new process to develop a logistic regression model to classify customers into those who are likely to accept personal loan and those who are not using all the available variables—except ID and ZIP Code — as predictors. However, this time modify the cutoff probability in such a way that the accuracy of identifying the positive responders is at least 70%. Compare the predictive accuracy of this revised model with that of the model developed in Part (2). (Again, try to be analytical instead of just noting the numbers)

```python

x1=df1.drop(['ID'],axis =1)
x1=df1.drop(['ZIP Code'],axis =1)
#drops ID and Zip code from test

decisions = (model.predict_proba(x_test)[:,1] >= 0.3).astype(int)
#gives smaller cutoff, vs. typical .5

tn, fp, fn, tp = confusion_matrix(np.array(y_test), decisions, labels=[0,1]).ravel()
#converts to string

print(tn)
print(fp)
print(fn)
print(tp)
score1=accuracy_score(y_test, decisions)
print(score1)
```

>tn = 3178

>fp = 0

>fn = 0

>tp = 322

>Accuracy = 100%


## 4. Aside from the problem of predicting the likelihood of accepting loan offers, think of two other business problems where logistic regressions can be utilized for predictive modeling. For each problem, identify a target variable and four possible predictor variables.
>Logistic regression could be utilized in many busniess applications. One example of where Logistic regression could be utilized is in the auto insurance industry. The target variable would be the monthly premium for a given consumer. Predictor variables could include: Prior tickets, Age, Credit Score, Gender, Vehical Price, Time with company, or Average distance driven per year. Another example of where logistic regression could be utilized is in predicting the probability of a presidential election. The predictor variables in this case could include: Incombant party, Amount of money spent on election, Average hours per week spent campainging, or Amount of money spent on negative advertisement. While both of these models are very different, both examples could be utilized for logistic regression modeling. 
