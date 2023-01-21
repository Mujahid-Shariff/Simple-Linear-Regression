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
df=pd.read_csv("delivery_time.csv")
df
df.head()
df.shape

#step 2 #split the variables in X and Y's
X=df[["Sorting Time"]]
#using [[]] for two dimention, as SKlearn's X variable in line 43, always uses two dimention data, only for simple linear regression.
Y=df[["Delivery Time"]]

#EDA Analysis]
#scatterplot
import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0], Y, color="red") #;,0 means only 0th column will be taken for scatterplot, 0th here stands for 1st column
plt.show()

#adding names to X an Y axis variables
import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0], Y, color="red")
plt.ylabel("Delivery Time") #naming the Y variable
plt.xlabel("Sorting Time") #naming the X variable
plt.show()

#model fitting - Bo  and B1
#Scikit learn - Package

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR

LR.fit(X,Y)

LR.intercept_ #Bo = C value
LR.coef_ #B1 = M(Slope) value


#getting the prediction value

Y_pred =LR.predict(X) #here X1 = our X variable, drug #.predict is used in python for calculating Bo and B1 for our X variable, variable might change based on our data

Y_pred

#calculating the scatter plot for original data and Y_pred values

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0], Y, color="red")
plt.scatter(X.iloc[:,0], Y_pred, color="Green") #.scatter for dots on the data
plt.ylabel("Delivery Time") #naming the Y variable
plt.xlabel("Sorting Time") #naming the X variable
plt.show()


import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0], Y, color="red")
plt.plot(X.iloc[:,0], Y_pred, color="Green") #.scatter for dots on the data
plt.ylabel("Delivery Time") #naming the Y variable
plt.xlabel("Sorting Time") #naming the X variable
plt.show()
#green is model predicted line, Y_pred
#red is actual data

from sklearn.metrics import mean_squared_error, r2_score
mse=mean_squared_error(Y, Y_pred)


import numpy as np
RMSE = np.sqrt(mse)
print("root mean square error :", RMSE.round(2))
print("R Square Value :",r2_score(Y,Y_pred).round(3)*100)

##############################################

#Question - 2 - Salary


#step 1 importing the data file
import pandas as pd
df=pd.read_csv("Salary_Data.csv")
df
df.head()
df.shape

#step 2 #split the variables in X and Y's
X=df[["YearsExperience"]]
#using [[]] for two dimention, as SKlearn's X variable in line 43, always uses two dimention data, only for simple linear regression.
Y=df[["Salary"]]

#EDA Analysis]
#scatterplot
import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0], Y, color="red") #;,0 means only 0th column will be taken for scatterplot, 0th here stands for 1st column
plt.show()

#adding names to X an Y axis variables
import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0], Y, color="red")
plt.ylabel("Salary") #naming the Y variable
plt.xlabel("YearsExperience") #naming the X variable
plt.show()

#model fitting - Bo  and B1
#Scikit learn - Package

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR

LR.fit(X,Y)

LR.intercept_ #Bo = C value
LR.coef_ #B1 = M(Slope) value


#getting the prediction value

Y_pred =LR.predict(X) #here X1 = our X variable, drug #.predict is used in python for calculating Bo and B1 for our X variable, variable might change based on our data

Y_pred

#calculating the scatter plot for original data and Y_pred values

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0], Y, color="red")
plt.scatter(X.iloc[:,0], Y_pred, color="Green") #.scatter for dots on the data
plt.ylabel("Salary") #naming the Y variable
plt.xlabel("YearsExperience") #naming the X variable
plt.show()


import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0], Y, color="red")
plt.plot(X.iloc[:,0], Y_pred, color="Green") #.scatter for dots on the data
plt.ylabel("Salary") #naming the Y variable
plt.xlabel("YearsExperience") #naming the X variable
plt.show()
#green is model predicted line, Y_pred
#red is actual data

from sklearn.metrics import mean_squared_error, r2_score
mse=mean_squared_error(Y, Y_pred)


import numpy as np
RMSE = np.sqrt(mse)
print("root mean square error :", RMSE.round(2))
print("R Square Value :",r2_score(Y,Y_pred).round(3)*100)

# Rsquare value is above than 90%.
# So, our model is Excellent.
############################################