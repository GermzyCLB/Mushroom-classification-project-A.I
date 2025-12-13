from sklearn.ensemble import RandomForestClassifier

# Random Forest hyperparamter grid
rf_param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 5],
    'max_features': ['sqrt', 'log2']
    }


