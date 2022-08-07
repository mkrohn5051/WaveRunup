# WaveRunup
Predicting wave runup heights as a result of a subduction-induced earthquake

One of my first Machine Learning projects done in graduate school at the University of Washington.  Really fun project with a heavy emphasis on feature engineering...with results that aren't too shabby!

This is the completed project file.  It shows the process of importing the data, understanding what I have, dealing with missing/duplicate/extra data, and finally drilling down into creating the model and applying it.

Dependencies:

Data source: the runups_distance.xslx file  

Python packages:  

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from datatile.summary.df import DataFrameSummary  
from sklearn import linear_model  
from sklearn.metrics import mean_absolute_error  
from sklearn.model_selection import cross_val_score  
from sklearn import datasets  
from sklearn import metrics  
from sklearn.metrics import roc_curve  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
from sklearn.model_selection import cross_val_score  
from sklearn import metrics  
import interpret  
from interpret.glassbox import ExplainableBoostingClassifier  
from interpret import show  
import warnings  
warnings.filterwarnings("ignore")  
from sklearn import svm  
from sklearn.svm import LinearSVC  
from sklearn.preprocessing import MinMaxScaler  
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import RobustScaler  
from sklearn.impute import SimpleImputer  
from sklearn import linear_model  
from sklearn.linear_model import LinearRegression  
from sklearn.linear_model import LassoCV  
from sklearn.linear_model import ElasticNetCV  
from sklearn.svm import SVR  
import interpret  
from interpret.glassbox import ExplainableBoostingClassifier  
from interpret import show  
import pycaret  
from pycaret.regression import *  # regression  
from lightgbm import LGBMRegressor  
