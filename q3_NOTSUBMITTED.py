# @author: Y384****

# Mark distribution for this task: 
# > Correctly producing first 3 scatterplots [5_marks]
# > Correctly producing the principal components scatterplot [5_marks]
# > Justification of: [10_marks total for this component of task]
    # > Choice of Classifier
    # > Choice of two variables to use
import sys, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.decomposition import PCA

from sklearn.linear_model import LassoCV, LinearRegression
linearRegModelObject = LinearRegression()

pca_ex = sys.argv[1]
classes = ""
# Below handles cases where secretfilepath doesn't exist!
if len(sys.argv) < 3:
    classes = pca_ex
else:
    classes = sys.argv[2]

data = np.loadtxt(pca_ex,delimiter=',')
X1 = data[:,np.newaxis, 0]
X2 = data[:,np.newaxis, 1]
X3 = data[:,np.newaxis, 2]

classes = np.loadtxt(classes,delimiter=' ')
y = classes

############################
############################
###OUTPUT SECTION FOLLOWS###
############################
############################

plt.xlabel("X1")
plt.ylabel("X2")
for p in range(len(y)):
    if y[p] == 1:
        plt.scatter(X1[p], X2[p], color='b',s=5)
    else:
    
        plt.scatter(X1[p], X2[p], color='r',s=5)
plt.show()

plt.xlabel("X1")
plt.ylabel("X3")
for p in range(len(y)):
    if y[p] == 1:
        plt.scatter(X1[p], X3[p], color='b',s=5)
    else:
    
        plt.scatter(X1[p], X3[p], color='r',s=5)
plt.show()

plt.xlabel("X2")
plt.ylabel("X3")
for p in range(len(y)):
    if y[p] == 1:
        plt.scatter(X2[p], X3[p], color='b',s=5)
    else:
    
        plt.scatter(X2[p], X3[p], color='r',s=5)
plt.show()

# # # # # # # # Display Scatterplot 1
# # # # # # # plt.scatter(X1, X2, edgecolor='b', s=20)
# # # # # # # plt.show()
# # # # # # # plt.close()
# # # # # # # # Display Scatterplot 2
# # # # # # # plt.scatter(X1, X3, edgecolor='b', s=20)
# # # # # # # plt.show()
# # # # # # # plt.close()
# # # # # # # # Display Scatterplot 3
# # # # # # # plt.scatter(X2, X3, edgecolor='r', s=20)
# # # # # # # plt.show()
# # # # # # # plt.close()

# 4th scatterplot to do
# Two axes are the first two principal components of data in pca_ex.csv 
# Differentiate these datapoints with different colours.

# What does this mean? :O 

pca = PCA(n_components=3)
pca.fit(data)

newX = pca.fit_transform(data)


pca2 = PCA(n_components=2)
pca.fit(data)
newX1 = pca.fit_transform(data)
# print(newX1)
# print(data - newX1)

# # plt.scatter(newX1[:,0],newX1[:,1])
# # plt.show()


plt.xlabel("PC1")
plt.ylabel("PC2")
for p in range(len(y)):
    if y[p] == 1:
        plt.scatter(newX1[:,0][p], newX1[:,1][p], color='b',s=5)
    else:
    
        plt.scatter(newX1[:,0][p], newX1[:,1][p], color='r',s=5)
plt.show()


# plt.xlabel("X2")
# plt.ylabel("X3")
# for p in range(len(y)):
    # if y[p] == 1:
        # plt.scatter(newX1[:,1][p], newX1[:,2][p], edgecolor='b',s=20)
    # else:
    
        # plt.scatter(newX1[:,1][p], newX1[:,2][p], edgecolor='r',s=20)
# plt.show()

# plt.xlabel("X1")
# plt.ylabel("X3")
# for p in range(len(y)):
    # if y[p] == 1:
        # plt.scatter(newX1[:,0][p], newX1[:,2][p], edgecolor='b',s=20)
    # else:
    
        # plt.scatter(newX1[:,0][p], newX1[:,2][p], edgecolor='r',s=20)
# plt.show()










