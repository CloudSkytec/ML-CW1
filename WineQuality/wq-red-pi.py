import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# read the CSV file
df = pd.read_csv('winequality-red.csv', delimiter=';')

# check dataset
df.head()
df.describe()
df.info()

# create box plots
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
        sns.distplot(value, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

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
top_features = importance_df.head(8)['Feature']
print(f"Top 5 Features: {list(top_features)}")

# Optional: You can retrain the model using only the top features to see if performance improves.
X_train_top = X_train[top_features]
X_valid_top = X_valid[top_features]

# Retrain the model using only the selected features
model_top = RandomForestClassifier(n_estimators=100, random_state=42)
model_top.fit(X_train_top, y_train)

# Re-evaluate the model
new_accuracy = model_top.score(X_valid_top, y_valid)
print(f"Accuracy with Top Features: {new_accuracy:.4f}")