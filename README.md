# WaveRunup
Predicting wave runup heights as a result of a subduction-induced earthquake

One of my first Machine Learning projects done in graduate school at the University of Washington.  Really fun project with a heavy emphasis on feature engineering...with results that aren't too shabby!

This is the completed project file.  It shows the process of importing the data, understanding what I have, dealing with missing/duplicate/extra data, and finally drilling down into creating the model and applying it.

Dependencies:

import pandas as pd\n
import numpy as np\n
import matplotlib.pyplot as plt\n
import seaborn as sns\n
from datatile.summary.df import DataFrameSummary\n
from sklearn import linear_model\n
from sklearn.metrics import mean_absolute_error\n
from sklearn.model_selection import cross_val_score\n
from sklearn import datasets\n
from sklearn import metrics\n
from sklearn.metrics import roc_curve\n
from sklearn.model_selection import train_test_split\n
from sklearn.metrics import confusion_matrix\n
from sklearn.model_selection import cross_val_score\n
from sklearn import metrics\n
import interpret\n
from interpret.glassbox import ExplainableBoostingClassifier\n
from interpret import show\n
import warnings\n
warnings.filterwarnings("ignore")\n
from sklearn import svm\n
from sklearn.svm import LinearSVC\n
from sklearn.preprocessing import MinMaxScaler\n
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
