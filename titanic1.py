# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:33:17 2020

@author: hp
"""

#  Importing the libraries

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report,roc_curve,auc

#  IMPORTING DATASETS

titanic_train=pd.read_csv("F:/Kaggle/Titanic/train.csv")
titanic_train.shape
titanic_train.info()
titanic_train['testing']=0
target=titanic_train['Survived']
ID_train=titanic_train['PassengerId']
titanic_train.drop("Survived", axis=1, inplace=True)

titanic_test=pd.read_csv("F:/Kaggle/Titanic/test.csv")
titanic_test.shape
titanic_test.info()
ID_test=titanic_test['PassengerId']
titanic_test['testing']=1

#   PREPROCESSING THE DATA

titanic=titanic_train.append(titanic_test)
titanic.shape
titanic.info()
titanic.drop("PassengerId", axis=1, inplace=True)
titanic.drop("Ticket", axis=1, inplace=True)
titanic.drop("Name", axis=1, inplace=True)
titanic.Pclass.value_counts()
titanic.Age.mean()


# Splitting data into X_train,X_test,Y_train

X_train=titanic[titanic.testing == 0]
X_train=X_train.drop(columns='testing')
Y_train=target
X_test=titanic[titanic.testing == 1]
X_test=X_test.drop(columns='testing')

# Data Cleansing for training test

X_train.Sex[X_train.Sex == 'male']=1
X_train.Sex[X_train.Sex == 'female']=0

X_train.Cabin[X_train.Cabin.isnull() == False]=1
X_train.Cabin[X_train.Cabin.isnull() == True]=0

X_train.Fare[X_train.Fare.isnull() == True]=X_train.Fare.mean()
X_train.Age[X_train.Age.isnull() == True]=X_train.Age.mean()

X_train.Embarked.value_counts()
X_train.Embarked[X_train.Embarked.isnull() == True]="S"



# Data Cleansing for testing set

X_test.Sex[X_test.Sex == 'male']=1
X_test.Sex[X_test.Sex == 'female']=0

X_test.Cabin[X_test.Cabin.isnull() == False]=1
X_test.Cabin[X_test.Cabin.isnull() == True]=0


X_test.Fare[X_test.Fare.isnull() == True]=X_test.Fare.mean()
X_test.Age[X_test.Age.isnull() == True]=X_test.Age.mean()

X_test.Embarked.value_counts()
X_test.Embarked[X_test.Embarked.isnull() == True]="S"

# Categorizing for training set

X_train['Embarked']=X_train['Embarked'].astype('category')
X_train['Cabin']=X_train['Cabin'].astype('category')
X_train['Sex']=X_train['Sex'].astype('category')
X_train['Pclass']=X_train['Pclass'].astype('category')

# Categorizing for test set

X_test['Embarked']=X_test['Embarked'].astype('category')
X_test['Cabin']=X_test['Cabin'].astype('category')
X_test['Sex']=X_test['Sex'].astype('category')
X_test['Pclass']=X_test['Pclass'].astype('category')

# Encoding for training set

X_train=pd.get_dummies(X_train, columns=['Embarked'], prefix=['Embarked'])

# Encoding for test set

X_test=pd.get_dummies(X_test, columns=['Embarked'], prefix=['Embarked'])


# Plotting Y_train

trn=Y_train.value_counts()
trn

trn.plot.bar(color=('g','r'), alpha=0.9)
plt.title("Bar Plot of Y_train")
plt.xlabel("Default")
plt.ylabel("Counts")
plt.show

#    PREPARING A MODEL USING LOGISTIC REGRESSION

clf=LogisticRegression(fit_intercept=True,C=1e15)
clf
clf.fit(X_train,Y_train)

Xtrain_pred=clf.predict(X_train)
Xtrain_pred
len(Xtrain_pred)

Xtrain_pred_prob=clf.predict_proba(X_train)[:,1]
Xtrain_pred_prob

Xtrain_pred_prob02=[1 if i>0.2 else 0 for i in Xtrain_pred_prob]
len(Xtrain_pred_prob02)
Xtrain_pred_prob02

Xtrain_pred_prob05=[1 if i>0.5 else 0 for i in Xtrain_pred_prob]
len(Xtrain_pred_prob05)
Xtrain_pred_prob05

Xtrain_pred_prob07=[1 if i>0.7 else 0 for i in Xtrain_pred_prob]
len(Xtrain_pred_prob07)
Xtrain_pred_prob07

Xtrain_pred_prob06=[1 if i>0.6 else 0 for i in Xtrain_pred_prob]
len(Xtrain_pred_prob06)
Xtrain_pred_prob06

from sklearn.metrics import confusion_matrix

conf_train02=confusion_matrix(Y_train,Xtrain_pred_prob02)
conf_train02
# accuracy for 0.2 = (366+290)/891=73.6%

conf_train05=confusion_matrix(Y_train,Xtrain_pred_prob05)
conf_train05
# accuracy for 0.5 = (469+243)/891=79.9% ;  higher accuracy than previous --> we are going in the right direction

conf_train07=confusion_matrix(Y_train,Xtrain_pred_prob07)
conf_train07
# accuracy for 0.7 = (529+171)/891=78.56% ; lower accuracy than previous --> we should backtrack

conf_train06=confusion_matrix(Y_train,Xtrain_pred_prob06)
conf_train06
# accuracy for 0.6 = (499+227)/891=81.5% ; highest accuracy --> 0.6 is chosen as the threshold

Y_pred=clf.predict(X_test)
Y_pred
len(Y_pred)
Y_pred.sum()

Y_pred_prob=clf.predict_proba(X_test)[:,1]
Y_pred_prob
len(Y_pred_prob)

Ypred_prob06=[1 if i>0.6 else 0 for i in Y_pred_prob]
Ypred_prob06

# Creating report for Logistic Regression

sub1=pd.DataFrame(ID_test)
sub1['Survived']=Ypred_prob06
sub1.info()

sub1.to_csv(r'F:\Kaggle\Titanic\submission1.csv', index=None,header=True)

#    trying Naive Bayes

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_sc=sc.fit_transform(X_train)
X_test_sc=sc.fit_transform(X_test)

# Fitting Classifier to Training set

from sklearn.naive_bayes import GaussianNB
NB=GaussianNB()
NB.fit(X_train,Y_train)

# Assessing Accuracy on Training

Y_pred_NBt=NB.predict(X_train)
cm_NB=confusion_matrix(Y_train,Y_pred_NBt)
cm_NB  
# accuracy= 78.67% ; lower accuracy than logistic regression

# Predicting Test set results

Y_pred_NB=NB.predict(X_test)
Y_pred_NB

# Creating report for Naive Bayes

sub2=pd.DataFrame(ID_test)
sub2['Survived']=Y_pred_NB
sub2.info()

sub2.to_csv(r'F:\Kaggle\Titanic\submission2.csv', index=None,header=True)


#     Trying K-Nearest Neighbour algorithm

from sklearn.neighbors import KNeighborsClassifier

neighbors = np.arange(19, 31) 
train_accuracy_knn = np.empty(len(neighbors)) 
  
# Loop over K values 
for i, k in enumerate(neighbors): 
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train, Y_train) 
      
    # Compute traning and test data accuracy 
    train_accuracy_knn[i] = knn.score(X_train, Y_train)  
  
# Generate plot 
 
plt.plot(neighbors, train_accuracy_knn, label = 'Training dataset Accuracy') 
plt.legend() 
plt.xlabel('no_neighbors') 
plt.ylabel('Accuracy') 
plt.show() 

# Fitting KNN to training test

knn = KNeighborsClassifier(n_neighbors=27) 
knn.fit(X_train, Y_train) 
knn.score(X_train,Y_train)
  
# Predict on dataset which model has not seen before 
Y_pred_knn=knn.predict(X_test)

# Creating report for K-Nearest Neighbor

sub3=pd.DataFrame(ID_test)
sub3['Survived']=Y_pred_knn
sub3.info()

sub3.to_csv(r'F:\Kaggle\Titanic\submission3.csv', index=None,header=True)

#      Trying Decision tree

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Fitting model on Training set

Tr=tree.DecisionTreeClassifier(max_depth=10)
TrFit=Tr.fit(X_train,Y_train)
TrFit

Tr1=tree.DecisionTreeClassifier(max_depth=5)
TrFit1=Tr1.fit(X_train,Y_train)
TrFit1

# Assessing Accuracy on Training

Y_pred_Trt=Tr.predict(X_train)
cm_Tr=confusion_matrix(Y_train,Y_pred_Trt)
cm_Tr  
# accuracy=92.7%

Y_pred_Trt1=Tr1.predict(X_train)
cm_Tr1=confusion_matrix(Y_train,Y_pred_Trt1)
cm_Tr1
# accuracy=84.62%

# Predicting Test set results

Y_pred_Tr=Tr.predict(X_test)
Y_pred_Tr

Y_pred_Tr1=Tr1.predict(X_test)
Y_pred_Tr1

# Creating report for decision tree

sub4=pd.DataFrame(ID_test)
sub4['Survived']=Y_pred_Tr
sub4.info()

sub4.to_csv(r'F:\Kaggle\Titanic\submission4.csv', index=None,header=True)

sub6=pd.DataFrame(ID_test)
sub6['Survived']=Y_pred_Tr1
sub6.info()

sub6.to_csv(r'F:\Kaggle\Titanic\submission6.csv', index=None,header=True)


#     Trying Random Forest

# Fitting Random Forest Classifier to the dataset 
# import the classifier 
from sklearn.ensemble import RandomForestClassifier
  
 # create classifier object 
random = RandomForestClassifier(n_estimators = 100, bootstrap = True, max_features = 'sqrt') 
  
# fit the classifier with x and y data 
RF=random.fit(X_train, Y_train)  
RF

# Predicting Test set results

Y_pred_RF=RF.predict(X_test)
Y_pred_RF
len(Y_pred_RF)

# Creating report for Random Forest

sub5=pd.DataFrame(ID_test)
sub5['Survived']=Y_pred_RF
sub5.info()

sub5.to_csv(r'F:\Kaggle\Titanic\submission5.csv', index=None,header=True)

