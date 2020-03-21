# @author: Y3848373

import sys, operator
import numpy as np
import pandas as pd
from math import log
import matplotlib.pyplot as plt
import matplotlib 
#from scipy.stats import beta
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, LinearRegression
#from sklearn.svm import l1_min_c
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


pca_ex = sys.argv[1]
classes = sys.argv[2]
data = pd.read_csv(pca_ex)
class_list = np.loadtxt(classes,delimiter=' ')
print(data.head())
print(data.shape)
print(class_list.shape)



scaler = StandardScaler()
scaler.fit(data)
X_scaled = scaler.transform(data)
print("after scaling minimum", X_scaled.min(axis=0))

pca = PCA()
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print("shape of X_pca is: ", X_pca.shape)




ex_variance = np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print("     X1        |      X2      |      X3")
print(ex_variance_ratio)
print("First two components {X1, X2} contribute to " + str(100*(ex_variance_ratio[0] + ex_variance_ratio[1])) + "% of total variance")
print("Thus, it is good enough to choose these two variables")


pca2 = PCA(n_components=2)
pca2.fit(data)
newX = pca.fit_transform(data)
print('Transformed data\n', newX)
print()
plt.scatter(newX[:,0], newX[:,1])
print('Principal component directions\n', pca.components_)
plt.show()

print("Transformed data centred if mean ~ zero:")
print(np.mean(newX[:,0]),np.mean(newX[:,1]))
x = pca.components_[0]
y = pca.components_[1]


print('Dot product of PC directions is:')
print(np.matmul(pca.components_[0],pca.components_[1]))
print('Size (L^2 norm) of 1st PC direction is')
print(np.linalg.norm(pca.components_[0]))
print('Size (L^2 norm) of 2nd PC direction is')
print(np.linalg.norm(pca.components_[1]))
from scipy.linalg import svd
U, d, V = svd(data,full_matrices=False)
print(U.shape,d.shape,V.shape)
D = np.diag(d) # d is vector, need 2x2 matrix with d on diag
print('Principal components:')
print(np.matmul(U,D))
print('Principal component directions:')
print(V)

poly_features = PolynomialFeatures(degree=4)
x = x.reshape(-1,1)
x_poly = poly_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)
print(r2)

plt.scatter(x, y, s=10)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred),key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x,y_poly_pred,color='m')
plt.show()



