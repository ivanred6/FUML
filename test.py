# @author: Y384****
import numpy
import numpy as np
np.zeros(12,)
res = 2+5+3+4+8+5+9+5
print(str(res))
exit
pandas --help
pip install pandas
import pytorch
from __future__ import print_function
import torch
x = torch.rand(5,3)
print(x)
torch.cuda.is_available()
print(648/2)
325/2
325*2
print(352/500)
print(500-352)
from yellowbrick.regressor import ResidualsPlot
pip
from yellowbrick.regressor import ResidualsPlot
visualiser = ResidualsPlot([1,3,4,2,3,4,2])
visualiser.poof()
visualiser = ResidualsPlot()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
contfp = continuous.csv
contfp = "continuous.csv"
data = pd.read_csv(contfp)
data.head()
data.columns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
Xs = data.drop(['Y'], axis=1)
y = data['Y'].values.reshape(-1,1)
print(y)
print(Xs)
lin_reg = LinearRegression()
MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)
mean_MSE = np.mean(MSEs)
print(mean_MSE)
print(MSEs)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
print(alpha)
for a in alpha:
print(type(a))
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
print(ridge_regressor)
ridge_regressor.fit(Xs, y)
print("hello world")
ridge_regressor.fit(Xs, y)
ridge_regressor.best_params_
ridge_regressor.best_score_
from sklearn.linear_model import Lasso
lasso = Lasso()
print(parameters)
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(Xs, y)
lasso.regressor.best_params_
lasso_regressor.best_params_
lasso_regressor.best_score_
from sklearn.metrics import r2_score
import readline
readline.write_history_file('output.txt')
