import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Loading dataset
# Dataset Link: https://archive.ics.uci.edu/dataset/73/mushroom
 
# fetch dataset
mushroom = fetch_ucirepo(id=73)
 
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
print(mushroom.metadata) 
  
# variable information 
print(mushroom.variables) 

# One-hpt encoding (missing values treated as category)
x_encoded = pd.get_dummies(X, drop_first=False)
