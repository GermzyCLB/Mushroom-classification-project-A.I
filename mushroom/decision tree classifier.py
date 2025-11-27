import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

#Load dataset


# Fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# Data (as pandas dataframes) 
X = mushroom.data.features #all categorical features
y = mushroom.data.targets['class'] #turns it into a simple 1d arrayof e/p


X_encoded = pd.get_dummies(X, drop_first=False)

#2  split into train/val/ test (70/15/15)
# deal with the test 15% and the 85% train+val

X_train_val, X_test , y_train_val , y_test = train_test_split(X_encoded,y,test_size=0.15, stratify=y, random_state=42)


#3  calculate value for second split

val_size=0.15
remaining_after_test=1-0.15 #(=0.85)

val_of_fraction_remaining= val_size/remaining_after_test


X_train, X_val, y_train, y_val =train_test_split(X_train_val, y_train_val, test_size=val_of_fraction_remaining,stratify =y_train_val,random_state=42)

#accuracy without learning anything
#overfitting detection

# Majority-class baseline
majority_class = y_train.mode()[0]   # simplest way to get most common class

print("Majority class:", majority_class)

# Predict majority for val/test
y_val_pred = [majority_class] * len(y_val)
y_test_pred = [majority_class] * len(y_test)

print("Validation accuracy:", accuracy_score(y_val, y_val_pred))
print("Test accuracy:", accuracy_score(y_test, y_test_pred))


cm_baseline = confusion_matrix(y_test, y_test_pred, labels=['e','p'])
disp = ConfusionMatrixDisplay(confusion_matrix = cm_baseline , display_labels=['edible','poisonous'])

disp.plot()
plt.title("tuned decision tree(Test set)")
plt.show()