# @author: Y3848373
import sys, csv, math
import numpy as np
from math import log
import matplotlib.pyplot as plt
import matplotlib 
# from scipy.stats import beta
from sklearn import linear_model
from sklearn.linear_model import LassoCV, LinearRegression, Lasso
from sklearn.svm import l1_min_c
linearRegModelObject = LinearRegression()
contfilepath = sys.argv[1]
secretfilepath = ""
r2score = 0.0
lasso = Lasso(random_state=0, max_iter=10000)

# If no secretfilepath in args, use continuous.csv 
if len(sys.argv) < 3:
    secretfilepath = contfilepath
else:
    print("ERROR: you have not specified a secret test csv. Program output may be erroneous!")
    secretfilepath = sys.argv[2]
continuous_datapoints = []
secrettest_datapoints = []

class DataPoint:
    def __init__(self, in_X1, in_X2, in_X3, in_X4, in_X5, in_X6, in_X7, in_X8, in_X9, in_X10, in_Y):
        self.x1 = in_X1
        self.x2 = in_X2
        self.x3 = in_X3
        self.x4 = in_X4
        self.x5 = in_X5
        self.x6= in_X6
        self.x7 = in_X7
        self.x8 = in_X8
        self.x9 = in_X9
        self.x10 = in_X10
        self.y = in_Y
    def toString(self):
        print("X1: ", self.x1)
        print("X2: ", self.x2)
        print("X3: ", self.x3)
        print("X4: ", self.x4)
        print("X5: ", self.x5)
        print("X6: ", self.x6)
        print("X7: ", self.x7)
        print("X8: ", self.x8)
        print("X9: ", self.x9)
        print("X10: ", self.x10)
        print("Y: ", self.y)
        


with open(contfilepath, newline='') as csvfile:
    counterx = 0
    filereader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in filereader:
        counterx += 1
        #print(', '.join(row)) # Debugging feature
        if counterx > 1:
            # x1_arr.append(row[0])
            # x2_arr.append(row[1])
            # x3_arr.append(row[2])
            # y_arr.append(row[3])
            continuous_datapoints.append( DataPoint(float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10])) )

# Start with continuous.csv
# Read in the predictors (X1, ... , X10) 
# store in continuous_data as a np.numpyarray? 
# We store Y values (target) as continuous_target which is a vector.

# np.genfromtxt(contfilepath,delimiter=',',missing_values='?')

for dp in continuous_datapoints:
    print(dp.toString())



with open(secretfilepath, newline='') as csvfile:
    counterx = 0
    filereader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in filereader:
        counterx += 1
        #print(', '.join(row)) # Debugging feature
        if counterx > 1:
            # x1_arr.append(row[0])
            # x2_arr.append(row[1])
            # x3_arr.append(row[2])
            # y_arr.append(row[3])
            secrettest_datapoints.append( DataPoint(float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), None) )

# Diagnostic code for secret datafile.
for sp in secrettest_datapoints:
    print(sp.toString())
    
    
data = np.loadtxt(contfilepath,delimiter=',')
print(type(data))
X1 = data[:,np.newaxis, 0]
X2 = data[:,np.newaxis, 1]
X3 = data[:,np.newaxis, 2]
X4 = data[:,np.newaxis, 3]
X5 = data[:,np.newaxis, 4]
X6 = data[:,np.newaxis, 5]
X7 = data[:,np.newaxis, 6]
X8 = data[:,np.newaxis, 7]
X9 = data[:,np.newaxis, 8]
X10 = data[:,np.newaxis, 9]

Xs = data[:,:10]
y = data[:,1]

print(Xs)
print(y)
plt.xlabel("Values for X variables")
plt.ylabel("Y")
plt.scatter(X1, y, edgecolor='b', s=20, label="Samples")
plt.scatter(X2, y, edgecolor='r', s=20, label="Samples")
plt.scatter(X3, y, edgecolor='g', s=20, label="Samples")
plt.scatter(X4, y, edgecolor='y', s=20, label="Samples")
# plt.scatter(X5, y, edgecolor='b', s=20, label="Samples")
# plt.scatter(X6, y, edgecolor='g', s=20, label="Samples")
# plt.scatter(X7, y, edgecolor='r', s=20, label="Samples")
# plt.scatter(X8, y, edgecolor='y', s=20, label="Samples")
# plt.scatter(X10, y, edgecolor='b', s=20, label="Samples")
# plt.scatter(X9, y, edgecolor='g', s=20, label="Samples")
plt.show()
plt.close()
linearRegModelObject.fit(X1,y)
linearRegModelObject.coef_
linearRegModelObject.intercept_

lasso.fit(Xs,y)
coefs = lasso.coef_
lasso_intercept = lasso.intercept_

print(coefs)
print(lasso_intercept)




continuous_data = Xs
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []

for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularisation')
plt.axis('tight')
plt.show()


test_samples = "Shape = (n_samples, n_features)"
true_values_forX = "true values for X"




secret_y_list = []
r2score = lasso.score(test_samples, true_values_forX)


############################
############################
###OUTPUT SECTION FOLLOWS###
############################
############################
# Outputs predicted Y value for each data point in secrettest.csv

# Outputs R2 value (r2_score) for fitted regression model when evaluated
# on secrettest.csv
print("R2 score: " + str(r2score))
print(secret_y_list)


# we want R2 to be as close to 1 as possible.

# print(r2score) # need to make r2score and understand how to use it properly! 








#############Code Dump

####
## File validation to ensure proper entry, may remove in final cut of
## code for submission
####
if secretfilepath != "secrettestset.csv":
    print("ERROR: file is not as specified in assessment rubric")
    print("File in question: secrettestset.csv != \'" + secretfilepath + "\'")

if contfilepath != "continuous.csv":
   print("ERROR: file is not as specified in assessment rubric")
   print("File in question: continuous.csv != \'" + contfilepath + "\'") 








def scatterplot(features, target):
    plt.figure(figsize=(16,8))
    plt.scatter(
        data[features],
        data[target],
        c='black'
    )
    plt.xlabel("X Features")
    plt.ylabel("Y Values")
    plt.show()




# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import sys, math, warnings
# from sklearn.model_selection import cross_val_score, GridSearchCV
# from  sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.metrics import r2_score
# warnings.simplefilter("ignore")
# contfilepath = sys.argv[1]
# secretpath = sys.argv[2]
# Y_output = []
# r2score = 0.0
# data = pd.read_csv(contfilepath)
# secrettest = pd.read_csv(secretpath)
# # print(data.head())
# # print(data.shape)
# Xs = data.drop(['Y'], axis=1)
# y = data['Y'].values.reshape(-1,1)
# secretXs = secrettest

# ### Regressions
# lin_reg = LinearRegression()

# MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)
# mean_MSE = np.mean(MSEs)
# #print("CV5 , LinReg: " + str(mean_MSE))

# MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=3)
# mean_MSE = np.mean(MSEs)
# #print("CV3 , LinReg: " + str(mean_MSE))

# MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=7)
# mean_MSE = np.mean(MSEs)
# #print("CV7 , LinReg: " + str(mean_MSE))

# alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 50, 100, 1000, 10000]
# ridge = Ridge()
# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 50, 100, 1000, 10000]}
# ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
# ridge_regressor.fit(Xs,y)
# # print(ridge_regressor.best_params_)
# # print(ridge_regressor.best_score_)
# # print("Above was cv5 ridge")
# # print("Now for cv3 ridge")
# ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=3)
# ridge_regressor.fit(Xs,y)
# # print(ridge_regressor.best_params_)
# # print(ridge_regressor.best_score_)
# # print("Now for cv7 ridge")
# ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=7)
# ridge_regressor.fit(Xs,y)
# # print(ridge_regressor.best_params_)
# # print(ridge_regressor.best_score_)

# lasso = Lasso()
# lasso_reg = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
# lasso_reg.fit(Xs,y)
# # print("Lasso regression now:")
# # print(lasso_reg.best_params_)
# # print(lasso_reg.best_score_)


# ### Predict the new Y_output
# print("Predicted Y values for secret test set:")
# lasout = lasso_reg.predict(secretXs)
# for yv in lasout:
    # print(yv)

# test = lasso_reg.predict(Xs)
# # print("### going to print the prediction of lasso, given Xs")
# # print(test)
# # print(y)
# #print("### going to try ridge, given Xs")
# # ri = ridge_regressor.predict(Xs)
# # print(ri)
# # print("### now secret ridge ")
# # risec = ridge_regressor.predict(secretXs)
# # print(risec)
# print("R^2 score (Lasso Regression):")
# r2score = r2_score(y, test)
# print(r2score)
# # print("now for ridge:")
# # print(r2_score(y, ri))
# # print("Now for linear")
# # lin_reg.fit(Xs, y)
# # lintest = lin_reg.predict(Xs)
# # print(r2_score(y, lintest))

# ## Code Dump






