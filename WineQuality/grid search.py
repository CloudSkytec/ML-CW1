import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.mstats import winsorize


# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 

# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets


# oversampling
from imblearn.over_sampling import SMOTE #conda install conda-forge::tpot-imblearn
oversample = SMOTE(k_neighbors=4)
X, y = oversample.fit_resample(X, y)
ys = y.values.ravel()
df = pd.concat([X, y], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, ys, test_size=0.2, random_state=42)

# Set up a pipeline to scale features and apply logistic regression
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('clf', LogisticRegression(max_iter=5000))
])

# Define the hyperparameter grid to search over
param_grid = {
    'clf__penalty': ['l1', 'l2', 'elasticnet', 'none'],       # Types of regularization
    'clf__C': [0.1, 1, 10, 100],                        # Regularization strength
    'clf__solver': ['saga', 'liblinear', 'lbfgs', 'sag'],     # Solvers for optimization
    'clf__l1_ratio': [0, 0.5, 1],                             # For elasticnet penalty, only used if 'penalty' is 'elasticnet'
}

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,                        # 5-fold cross-validation
    scoring='accuracy',          # Or use other metrics like 'f1', 'roc_auc' based on requirements
    n_jobs=-1,                   # Use all available cores
    verbose=2                    # Show progress
)

# Fit GridSearchCV
df.describe()

grid_search.fit(X_train, y_train)

# Get the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Evaluate on test set
test_score = grid_search.score(X_test, y_test)
print("Test Set Score:", test_score)