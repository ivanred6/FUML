# @author: Y384****
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, math, warnings
from sklearn.model_selection import cross_val_score, GridSearchCV
from  sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import r2_score
warnings.simplefilter("ignore")
contfilepath = sys.argv[1]
secretpath = sys.argv[2]
Y_output = []
r2score = 0.0
data = pd.read_csv(contfilepath)
secrettest = pd.read_csv(secretpath)
# print(data.head())
# print(data.shape)
Xs = data.drop(['Y'], axis=1)
y = data['Y'].values.reshape(-1,1)
secretXs = secrettest

### Regressions
lin_reg = LinearRegression()
logit = LogisticRegression()

MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)
mean_MSE = np.mean(MSEs)
print("CV5 , LinReg: " + str(mean_MSE))

MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=3)
mean_MSE = np.mean(MSEs)
print("CV3 , LinReg: " + str(mean_MSE))

MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=7)
mean_MSE = np.mean(MSEs)
print("CV7 , LinReg: " + str(mean_MSE))

alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 50, 100, 1000, 10000]
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 50, 100, 1000, 10000]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(Xs,y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
print("Above was cv5 ridge")
print("Now for cv3 ridge")
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=3)
ridge_regressor.fit(Xs,y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
print("Now for cv7 ridge")
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=7)
ridge_regressor.fit(Xs,y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

lasso = Lasso()
lasso_reg = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_reg.fit(Xs,y)
print("Lasso regression now:")
print(lasso_reg.best_params_)
print(lasso_reg.best_score_)



### Predict the new Y_output
print("Predicted Y values for secret test set:")
lasout = lasso_reg.predict(secretXs)
#print(lasout)

for yv in lasout:
    print(yv)

test = lasso_reg.predict(Xs)
print("### going to print the prediction of lasso, given Xs")
print(test)
print(y)
print("### going to try ridge, given Xs")
ri = ridge_regressor.predict(Xs)
print(ri)
print("### now secret ridge ")
risec = ridge_regressor.predict(secretXs)
print(risec)
print("R^2 score (Lasso Regression):")
r2score = r2_score(y, test)
print(r2score)
print("now for ridge:")
print(r2_score(y, ri))
print("Now for linear")
lin_reg.fit(Xs, y)
lintest = lin_reg.predict(Xs)
print(r2_score(y, lintest))

# Code Dump






