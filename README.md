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

![image](https://user-images.githubusercontent.com/61456930/76241236-09650880-620b-11ea-8fbc-1aab18adbdc4.png)

```python
import pandas as pd
import seaborn as sns
#imports the graphing program seaborn that i will be using
df = pd.read_csv("/Users/Charles/Desktop/GMU/GBUS738/UniversalBank.csv", delimiter="," , header = 3, skiprows=0)
#header is on the 4th row of my csv file, counting this goes row 0, row 1, row 2, and finally row 3. This means for header we set it = to 3
sns.scatterplot('Age', 'Income',hue='Personal Loan', data=df ).set(title = 'Acceptance of Personal Loans', xlabel = 'Age in Years', ylabel = 'Income in $1,000s')
```

>Based on the information from UniversalBank, and our scatterplot, it does appear there is a stronger correlation between Income and loan acceptance, as compared to age and loan acceptance. Based on visually looking at our scatter plot, we can see that plot points listed in orange represent an individual taking a given loan, where blue represents those not taking the loan. While we can see there is a spread across all age groups of those taking or not taking loans, we can see a stronger correlation between income and those who accepted loans. It does appear that those individuals who have an income of >$100,000, are more strongly correlated with loan acceptance, as compared to those who have an income of <$100,000. Based on the graph, those who have an income sub $100,000, had very few loans, where as the majority of loans were above the income range of $100,000. 

## 2. Build a logistic regression model to classify customers into those who are likely to accept personal loan offer and those who are not. Use all the available variables as predictors except ID and ZIP Code. (Hint: Since the Logistic Regression operator expects binominal or polynominal target variables, if the target variable is numeric, you will have to convert it to binominal by using the Numerical to Binominal operator.)

 - a. Evaluate the overall predictive accuracy of the model as well as the accuracy of each class
using appropriate metrics.

 - b. What was the default cutoff probability used to generate the classifications?

-  c. Assuming that the dataset contains a representative sample of the liability customers of
the bank, if you target 100 customers randomly (i.e., without the aid of any predictive
model), how many of them would potentially accept a personal loan offer?

-  d. Now if you use your model in Part (2) to select 100 customers with the highest probability
of loan acceptance, how many of them would potentially accept a personal loan offer?
(Hint: Revise the process from Part (2) to generate a lift chart.)

 - e. What percentage of customers who accepted the loan were incorrectly classified by the
model in Part (2)?

## 3. Suppose the bank is interested in improving the accuracy of identifying the potential positive
responders, i.e., those who would accept the loan offer. Create a new process to develop a
logistic regression model to classify customers into those who are likely to accept personal loan
and those who are not using all the available variables—except ID and ZIP Code — as predictors.
However, this time modify the cutoff probability in such a way that the accuracy of identifying
the positive responders is at least 70%. Compare the predictive accuracy of this revised model
with that of the model developed in Part (2). (Again, try to be analytical instead of just noting the
numbers)

## 4. Aside from the problem of predicting the likelihood of accepting loan offers, think of two other
business problems where logistic regressions can be utilized for predictive modeling. For each
problem, identify a target variable and four possible predictor variables.
