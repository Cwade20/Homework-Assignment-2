# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:31:00 2020

@author: Nikhith
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
df=pd.read_excel(r'C:\Users\Owner\Downloads\UniversalBank.xlsx',sheet_name = 1,header = 3)

plt.figure(figsize=(18, 6))
sns.set_context("notebook", font_scale=1.0)
graph=sns.scatterplot(x="Age",y="Income",hue="Personal Loan",data=df)
plt.title("Scatter plot of Age vs Income")
plt.ylabel("Income(in $1000's)")
plt.legend((0, 1), ('label1', 'label2'))


#Generating the model
X = df.iloc[:,[1,2,3,5,6,7,8,10,11,13]]
y = df.iloc[:,[9]]

lr = LogisticRegression()

model = lr.fit(X, y)
preds = model.predict(X)
cm=confusion_matrix(y, preds)
print(cm)
scores = classification_report(y, preds, output_dict = True)
def accuracies(cm):
    accuracy =((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))*100
    print('overall accuracy = ',accuracy)

    accuracy_class0=(cm[0][0]/(cm[0][0]+cm[0][1]))*100
    print('accuracy of class0 = ',accuracy_class0)

    accuracy_class1=(cm[1][1]/(cm[1][1]+cm[1][0]))*100
    print('accuracy of class1 = ',accuracy_class1)
accuracies(cm)

#lift chart
predicted_probas = lr.predict_proba(X)
import scikitplot as skplt
skplt.metrics.plot_cumulative_gain(y, predicted_probas)
skplt.metrics.plot_lift_curve(y, predicted_probas)

# model after removing some variables
X_new = df.iloc[:,[1,2,3,5,6,7,10,11,13]]
y_new = df.iloc[:,[9]]

lr_new = LogisticRegression()

model_new = lr_new.fit(X_new, y_new)
preds_new = model_new.predict(X_new)
cm_new = confusion_matrix(y_new, preds_new)
print(cm_new)
scores_new = classification_report(y_new, preds_new, output_dict = True)
accuracies(cm_new)

# scores after changing the cut off probability
predicted_probas_new = lr_new.predict_proba(X_new)[:,1]    
predicted_new=[1 if i > 0.25 else 0 for i in predicted_probas_new]
cm_prob = confusion_matrix(y_new,predicted_new)
print(cm_prob)
scores_n = classification_report(y_new, predicted_new, output_dict = True)
accuracies(cm_prob)
