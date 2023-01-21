# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:32:36 2022

@author: admin
"""
#@Mujahid Shariff

#Question 1 - Delivery time

#Simple linear regression

#step 1 importing the data file
import pandas as pd
import numpy as np

df=pd.read_csv("delivery_time.csv")
df
df.head()
df.shape

#step 2 #split the variables in X and Y's using logrithm transformation
X_log = np.log(df['Sorting Time'].values.reshape(-1,1))
#using [[]] for two dimention, as SKlearn's X variable in line 43, always uses two dimention data, only for simple linear regression.
y_log = df['Delivery Time'].values.reshape(-1,1)

#model fitting and finding out intercept and Co-efficient values
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
X_train_log, X_test_1og, Y_train_log, Y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state= 42)

#LR fit
y_pred_log= LinearRegression()
y_pred_log.fit(X_train_log,Y_train_log)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

print(" Intercept value of Model is " ,y_pred_log.intercept_)
print("Co-efficient Value of Log Model is : ", y_pred_log.coef_)

l_model= y_pred_log.predict(X_test_1og)
l_model

SLR = pd.DataFrame({'Actual':Y_test_log.flatten(), 'Predict': l_model.flatten()})
SLR

import matplotlib.pyplot as plt
#calculating the rate of error using scatterplot
plt.scatter(X_test_1og, Y_test_log,  color='gray')
plt.plot(X_test_1og, l_model, color='red', linewidth=2)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test_log, l_model))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test_log, l_model) ) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test_log, l_model)))
print("R^2 Score :          ", metrics.r2_score(Y_test_log, l_model))

#metrics.r2_score(Y_test_log, l_model))
#Mean Absolute Error: 1.91
#Mean Squared Error: 6.38
#Root Mean Squared Error: 2.52
#R^2 Score :           0.080


#Question - 2 - Salary
#step 1 importing the data file
import pandas as pd
df=pd.read_csv("Salary_Data.csv")
df
df.head()
df.shape

#step 2 #split the variables in X and Y's using logrithm transformation
X_log = np.log(df['YearsExperience'].values.reshape(-1,1))
#using [[]] for two dimention, as SKlearn's X variable in line 43, always uses two dimention data, only for simple linear regression.
y_log = df['Salary'].values.reshape(-1,1)

#model fitting and finding out intercept and Co-efficient values
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
X_train_log, X_test_1og, Y_train_log, Y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state= 42)

#LR fit
y_pred_log= LinearRegression()
y_pred_log.fit(X_train_log,Y_train_log)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

print(" Intercept value of Model is " ,y_pred_log.intercept_)
print("Co-efficient Value of Log Model is : ", y_pred_log.coef_)

l_model= y_pred_log.predict(X_test_1og)
l_model

SLR2 = pd.DataFrame({'Actual':Y_test_log.flatten(), 'Predict': l_model.flatten()})
SLR2

import matplotlib.pyplot as plt
#calculating the rate of error using scatterplot
plt.scatter(X_test_1og, Y_test_log,  color='gray')
plt.plot(X_test_1og, l_model, color='red', linewidth=2)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test_log, l_model))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test_log, l_model) ) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test_log, l_model)))
print("R^2 Score :          ", metrics.r2_score(Y_test_log, l_model))

#metrics.r2_score(Y_test_log, l_model))
#Mean Absolute Error: 7714.638101223965
#Mean Squared Error: 84567607.74831854
#Root Mean Squared Error: 9196.064796874723
#R^2 Score :           0.8344395471211845
############################################