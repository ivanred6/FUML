# @author: Y384****
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

%matplotlib inline
contfilepath = sys.argv[1]
secretfilepath = ""
# Below handles cases where secretfilepath doesn't exist!
if len(sys.argv) < 3:
    secretfilepath = contfilepath
else:
    secretfilepath = sys.argv[2]
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
   
DATAPATH = contfilepath
data = pd.read_csv(DATAPATH)
data.head()
