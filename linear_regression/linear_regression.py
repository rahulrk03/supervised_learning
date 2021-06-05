# -*- coding: utf-8 -*-

from google.colab import files
  
uploaded = files.upload()

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split   # Sklearn package's randomized data splitting function
import io

cData = pd.read_csv(io.BytesIO(uploaded['auto-mpg.csv']))
cData.head()

# dropping/ ignoring car_name
cData = cData.drop('car name', axis=1)
# Also replacing the categorical var with actual values
cData['origin'] = cData['origin'].replace({1: 'America', 2: 'Europe', 3: 'Asia'})
cData.head()

# Create Dummy Variables
cData = pd.get_dummies(cData, columns=['origin'])
cData.head()

#A quick summary of the data columns
cData.describe() # applicable for all numerical columns

# hp is missing cause it does not seem to be recognized as a numerical column!
cData.dtypes

# isdigit()? on 'horsepower'
hpIsdigit = pd.DataFrame(cData.horsepower.str.isdigit())
cData[hpIsdigit['horsepower']==False]

# Missing values have a '?'
# Replace missing value with NaN
cData = cData.replace('?', np.nan)
cData[hpIsdigit['horsepower'] == False]

cData.median()

medianFiller = lambda x: x.fillna(x.median())
# This will replace every column's missing value with that column's median
cData = cData.apply(medianFiller, axis=0) # axis = 0 means columnwise.
cData['horsepower'] = cData['horsepower'].astype('float64')

"""
Bivariate plots
A bivariate analysis among the different varibales can be done using scatter matrix plot. Seaborn libs create a dashboard 
reflecting useful information about the dimensions. The result can be stored as a .png file
"""

cData_attr = cData.iloc[:, 0:7]
sns.pairplot(cData_attr, diag_kind="kde")  # to plot density curve instead of histogram on the diag

# Split Data
# lets build our linear model
#independent variables
X = cData.drop(['mpg', 'origin_Europe'],axis=1)
# depenedent variable
y = cData['mpg']

# Split X and y into training and test set in 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# Fit Liner Model

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Here are the coefficients for each variable and the intercept

for idx, col_name in enumerate(X_train.columns):
  # print(idx)
  # print((regression_model.coef_[])[idx])
  print("The coefficient for {} is {} ".format(col_name, regression_model.coef_[idx]))

intercept  = regression_model.intercept_ # in video lecture its given intercept_[0]
print("The intercept is {}".format(intercept))

# The score (R^2)  for in-sample and out of sample
regression_model.score(X_train, y_train)

# Out of sample score (R^2)
regression_model.score(X_test, y_test)

# Adding interaction terms

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train2 = poly.fit_transform(X_train)
X_test2 = poly.fit_transform(X_test)

poly_clf = linear_model.LinearRegression()
poly_clf.fit(X_train2, y_train)
y_pred = poly_clf.predict(X_test2)
# print(y_pred)

print(poly_clf.score(X_train2, y_train))

print(poly_clf.score(X_test2, y_test))

# this improves as the cost of 29 extra variables
print(X_train.shape)
print(X_train2.shape)

# Polynomianl features ( with only interaction terms) have improved the OUT off sample R^2. 
# However at the cost of increasing the number of variables significantly
