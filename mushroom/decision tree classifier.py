import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

#Load dataset

#1)load+preprocess for first time

# Fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# Data (as pandas dataframes) 
X = mushroom.data.features #all categorical features
# y = mushroom.data.targets['class'] #turns it into a simple 1d arrayof e/p-..was having issues with this so using .iloc[:,0] instead 
#as it doesnt matter if its a series or not.

y = mushroom.data.targets.iloc[:, 0]
 


X_encoded = pd.get_dummies(X, drop_first=False)

# split into train/val/ test (70/15/15)
# deal with the test 15% and the 85% train+val

X_train_val, X_test , y_train_val , y_test = train_test_split(X_encoded,y,test_size=0.15, stratify=y, random_state=42)


#pre-calculate value for second split

val_size=0.15
remaining_after_test=1-0.15 #(=0.85)

val_of_fraction_remaining= val_size/remaining_after_test


X_train, X_val, y_train, y_val =train_test_split(X_train_val, y_train_val, test_size=val_of_fraction_remaining,stratify =y_train_val,random_state=42)

#accuracy without learning anything
#overfitting detection


# 1)Majority-class baseline-have baseline fpor accuracy
majority_class = y_train.mode()[0]   # simplest way to get most common class

print("Majority class:", majority_class)

# Predict majority for val/test
y_val_pred = [majority_class] * len(y_val)
y_test_pred = [majority_class] * len(y_test)

print("Validation accuracy:", accuracy_score(y_val, y_val_pred))
print("Test accuracy:", accuracy_score(y_test, y_test_pred))

#now we have the original 52% reference point

cm_baseline = confusion_matrix(y_test, y_test_pred, labels=['e','p'])
disp = ConfusionMatrixDisplay(confusion_matrix = cm_baseline , display_labels=['edible','poisonous'])

disp.plot()
plt.title("test set from tuned decision tree:")
plt.show()


#defining the hyperparamater grid...defines tuneing procedure

param_grid = {
    'max_depth':[3,5,10,15,20,None], #controls size of the tree to prevent overfitting
    'min_samples_split':[2,5,10,20], #can create  arbitrarily small leaves
    'min_samples_leaf':[1,2,5,10], # garuntees each leaf has a minimum size,which avoids low variance.
    'criterion': ['gini','entropy']  #criterion tells decision tree how to measure the quality of the split when choosing what feature to split 
    
    
     }



 #to ensure that in each fold it is balanced in cross validation
cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


#function to run Grid search 
#runs twice(one oon real labels and one on shuffled labels as instructed to do in feedback )

def start_grid_search(X,y):
    model= DecisionTreeClassifier(random_state=42)
# creation of base decison tree model in which grid search cross validation will tune 
# by trying out different hyperparamaters.

    grid = GridSearchCV(model,param_grid, scoring= "f1_macro",cv = cv ,n_jobs = -1)
#fits the whole grid search on the training data 

    grid.fit(X,y)

#returns the whole GridSearchCVobject that was create above already
#containing the best model + results 
    return grid

 
 #now we need to able to see the hyperparamaters in each  iteration


#tunes the real labels in dataset as hyperparamaters are seleced automatically
grid_real_output = start_grid_search(X_train,y_train)


#prints out all hyperparamater combinations

print("\nBest paramaters gathered (REAL LABEL TERMS)",grid_real_output.best_params_)
print("\nBest cross validation f1-macro score(REAL)", grid_real_output.best_score_)


#next we need to inspect the repsective hyperparamater combinaions to make note of tuneing process in  ongoing notes.

cv_results_df = pd.DataFrame(grid_real_output.cv_results_)


#shows the paramaters ,mean test score,rank and sort it by the best rank

print("\n  === all grid search results(sorted by rank)===")

 
print(
cv_results_df[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
].sort_values('rank_test_score')


)


#output the best tuned decision tree results from the gridsearch cross validation
best_dct_real_output = grid_real_output.best_estimator_

#now evaluate the tuned model on he validation set 
y_val_pred_real = best_dct_real_output.predict(X_val)


print("\n === tuned decision tree(with real labels) alongside validation ===")

print("\nThe accuracy obtained:", accuracy_score(y_val, y_val_pred_real))

print(classification_report(y_val, y_val_pred_real))

#retrain it on the training + validation set and then do more testing

best_dct_final_real_output= DecisionTreeClassifier(**grid_real_output.best_params_,random_state=42)

best_dct_final_real_output.fit(X_train_val,y_train_val)



y_test_pred_real_lab_ =  best_dct_final_real_output.predict(X_test)


#print the results for the retrained decision tree of predicted values

print("\n=== Tuned decision tree(REAL Labels)-Test===")
print("Accuracy:",accuracy_score(y_test, y_test_pred_real_lab_))

print(classification_report(y_test, y_test_pred_real_lab_))




# Confusion matrix for tuned Decision Tree on test set
cm_dt = confusion_matrix(y_test, y_test_pred_real_lab_, labels=['e','p'])

disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt,
                             display_labels=['edible', 'poisonous'])
disp_dt.plot()
plt.title("Tuned Decision Tree – Test set")
plt.show()


