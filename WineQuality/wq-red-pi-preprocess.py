import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm
from sklearn import metrics

# read the CSV file
df = pd.read_csv('winequality-red.csv', delimiter=';')

# check dataset
df.head()
df.describe()
df.info()

# Create a new feature: Ratio of free sulfur dioxide to total sulfur dioxide
df['sulfur_ratio'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
# Drop the original features if they're no longer needed
df = df.drop(columns=['free sulfur dioxide', 'total sulfur dioxide'])

# todo outliers treatment
df['residual sugar_winsorized'] = winsorize(df['residual sugar'], limits=[0.01, 0.01])
df = df.drop(columns=['residual sugar'])

# Apply log transformation (add a small constant to avoid log(0))
df['sulphates'] = np.log(df['sulphates'] + 1e-6)
df['residual sugar_winsorized'] = np.log(df['residual sugar_winsorized'] + 1e-6)


# Selecting features for scaling, excluding the target 'quality'
features_to_scale = df.columns.drop('quality')

# Initialize Min-Max scaler and fit-transform data
min_max_scaler = MinMaxScaler()
df[features_to_scale] = min_max_scaler.fit_transform(df[features_to_scale])

# Display Min-Max scaled data
print("Min-Max Scaled Data:")
print(df)

# Assume df is your DataFrame with the 12 features and 'quality' as the target column
X = df.drop(columns=['quality'])
y = df['quality']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest classifier (you can use any model here)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the baseline performance on the validation set
baseline_accuracy = model.score(X_valid, y_valid)
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

# Apply Permutation Importance
perm_importance = permutation_importance(model, X_valid, y_valid, n_repeats=10, random_state=42)

# Get importance scores
importance_scores = perm_importance.importances_mean
feature_names = X.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

# Select the top N features (e.g., top 5)
n = 8
top_features = importance_df.head(n)['Feature']
print(f"Top {n} Features: {list(top_features)}")

# Optional: You can retrain the model using only the top features to see if performance improves.
X_train_top = X_train[top_features]
X_valid_top = X_valid[top_features]

# Retrain the model using only the selected features
model_top = RandomForestClassifier(n_estimators=200, random_state=42)
model_top.fit(X_train_top, y_train)

# Re-evaluate the model
new_accuracy = model_top.score(X_valid_top, y_valid)
print(f"Accuracy with Top Features: {new_accuracy:.4f}")

# KFold cross-validator
kf = KFold(n_splits=8, shuffle=True, random_state=42)
scores = cross_val_score(model_top, X, y, cv=kf, scoring='accuracy')
print("k-fold cross valuation:")
print("Scores for each fold:", scores)
print("Mean accuracy:", scores.mean())


# box plots
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    if col != 'type':
        sns.boxplot(y=col, data=df, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    if col != 'type':
        sns.histplot(value, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)