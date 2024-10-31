#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import random
import seaborn as sns
import scipy
from scipy import stats

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 

# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 


# Access the dataset's metadata
print(wine_quality.metadata.uci_id)
print(wine_quality.metadata.num_instances)
print(wine_quality.metadata.additional_info.summary)

# Access features and targets
features_df = df(wine_quality.data.features)
targets_df = df(wine_quality.data.targets)

# Combine into a DataFrame
data_df = pd.concat([features_df, targets_df], axis=1)
data_df.head()

data_features = data_df.drop(columns=["quality"])
data_targets = data_df["quality"]

# information of each feature
data_df.info()
data_df.describe()
data_df.head()


data_df["quality"].value_counts()
sns.countplot(x="quality",data = data_df)


# single feature boxplot(include outliers)

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20,15))

for i, column in enumerate(data_df.columns[:]):
    ax = axes[i // 4, i % 4]  # column index
    ax.boxplot(data_df[column], patch_artist=True)
    ax.set_title(column)
    ax.set_ylabel("Values")

plt.tight_layout()

data_df['quality'].value_counts()


# correlation matrix
data_corrMatt = data_df.corr(numeric_only=True)

# Generate a mask for the upper triangle
mask = np.zeros_like(data_corrMatt)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(20, 12))
plt.title("Wine Quality Correlation")

# Generate a custom diverging colormap
cmap = sns.diverging_palette(260, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(data_corrMatt, vmax=1.2, square=False, cmap=cmap, mask=mask, 
ax=ax, annot=True, fmt=".2g", linewidths=1);


# Define a function to identify outliers using the IQR method
def identify_outliers_iqr(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

# Create a DataFrame to store deleted outliers
deleted_outliers = df(columns=data_df.columns.tolist() + ['Outlier_Feature'])

# Identify outliers and remove them
for feature in data_df.columns[:-1]:
    lower_bound, upper_bound = identify_outliers_iqr(data_df, feature)
    
    # Find samples that are outliers and have moderate quality
    if feature == 'citric_acid':
        outliers = data_df[(data_df[feature] > upper_bound)]
    if feature == 'residual_sugar':
        outliers = data_df[(data_df[feature] > 40)]
    else:
        outliers = data_df[(data_df[feature] < lower_bound) | (data_df[feature] > upper_bound)]
        
   
    
    quality_condition = (outliers['quality'] >= 4) & (outliers['quality'] <= 7)
    
      # Define specific conditions for density, residual sugar, and alcohol
    
    # Filter out samples with moderate quality values (4-7)
    outliers_with_quality_condition = outliers[quality_condition]
    
     # Add outliers to the deleted_outliers DataFrame and mark the outlier feature
    if not outliers_with_quality_condition.empty:
        outliers_with_quality_condition = outliers_with_quality_condition.copy()
        outliers_with_quality_condition['Outlier_Feature'] = feature
        deleted_outliers = pd.concat([deleted_outliers, outliers_with_quality_condition])
    
    # Remove these samples from the dataset
    data_df = data_df[~data_df.index.isin(outliers[quality_condition].index)]


print(deleted_outliers)


# single feature boxplot(no outliers)

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20,15))

for i, column in enumerate(data_df.columns[:]):
    ax = axes[i // 4, i % 4]  # column index
    ax.boxplot(data_df[column], patch_artist=True)
    ax.set_title(column)
    ax.set_ylabel("Values")

plt.tight_layout()


# z-score normalization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_df)


# Combine free_sulfur_dioxide and total_sulfur_dioxide
data_combined = data_df

data_combined['free_sulfur_ratio'] = data_df['free_sulfur_dioxide'] / data_df['total_sulfur_dioxide']
data_combined = data_combined.drop(columns=['total_sulfur_dioxide','free_sulfur_dioxide'])

# correlation matrix
data_corrMatt = data_combined.corr(numeric_only=True)

# Generate a mask for the upper triangle
mask = np.zeros_like(data_corrMatt, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(20, 12))
plt.title("Wine Quality Correlation")

# Generate a custom diverging colormap
cmap = sns.diverging_palette(260, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(data_corrMatt, 
            vmax=1.2, 
            square=False, 
            cmap=cmap, 
            mask=mask, 
            ax=ax, 
            annot=True,          # Display the values in the heatmap
            fmt=".2f",          # Format the numbers (2 significant digits)
            linewidths=1, 
            cbar_kws={"shrink": .8})  # Optional: adjust color bar size

# Use SMOTE to deal with the imbalanced target data

from imblearn.over_sampling import SMOTE

X = data_combined.drop(columns=['quality'])  # Features
y = data_combined['quality']  # Target variable

# Apply SMOTE
smote = SMOTE(random_state=42,k_neighbors=3)

X_resampled, y_resampled = smote.fit_resample(X, y)

data_resampled = df(X_resampled, columns=X.columns)
data_resampled['quality'] = y_resampled

# single feature boxplot(include outliers)

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20,15))

for i, column in enumerate(data_resampled.columns[:]):
    ax = axes[i // 4, i % 4]  # column index
    ax.boxplot(data_resampled[column], patch_artist=True)
    ax.set_title(column)
    ax.set_ylabel("Values")

plt.tight_layout()


data_features1 = data_resampled.drop(columns=["quality"])
data_targets1 = data_resampled["quality"]

# split the data sets: traning set - 80%, validation set - 20%
fea_train, fea_val, tar_train, tar_val = train_test_split(data_features1, data_targets1, test_size=0.2, random_state=42, stratify=data_targets1)




# Feature importance based on mean decrease in impurity

# Define the Random Forest model
rf = RandomForestClassifier(random_state=0,n_estimators=300,max_depth=20,class_weight='balanced')
rf.fit(fea_train, tar_train)

# Calculate feature importances
importances = rf.feature_importances_

mdi_importance = pd.DataFrame({
    "Feature": data_features1.columns,
    "Importance": importances
}).sort_values(by='Importance', ascending=False)

# Std of feature importances
std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)

# Barplot of feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Feature", y="Importance", data=mdi_importance,yerr=std, capsize=.2)
plt.title("MDI Importance Barplot With Error Bar")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=-30)
plt.show()

# Calculate the accuracy on training set
train_pred = rf.predict(fea_train)
train_accuracy = accuracy_score(tar_train, train_pred)

# Calculate the accuracy on validation set
val_pred = rf.predict(fea_val)
val_accuracy = accuracy_score(tar_val, val_pred)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Sort by importance and select the top 5 features
top_features = mdi_importance.head(5)
selected_features = top_features['Feature'].tolist()
print("Selected Top 3 Features:", selected_features)

# Calculate accuracy on validation set
fea_val_selected = fea_val[selected_features]
rf_val = RandomForestClassifier(random_state=1,n_estimators=300,max_depth=20,class_weight='balanced')
rf_val.fit(fea_train[selected_features], tar_train)
tar_val_pred = rf_val.predict(fea_val_selected)
val_accuracy = accuracy_score(tar_val, tar_val_pred)

print(f"MDI Validation accuracy with top 5 features: {val_accuracy:.2f}")

# KFold cross-validator
from sklearn.model_selection import KFold, cross_val_score

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=2)
scores = cross_val_score(rf_val, data_features1, data_targets1, cv=kf, scoring='accuracy')
print("k-fold cross valuation:")
print("Scores for each fold:", scores)
print("Mean accuracy:", scores.mean())




# Feature importance based on feature permutation
start_time = time.time()
fp = permutation_importance(
    rf, fea_val, tar_val, n_repeats=10, random_state=2)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
fp_importances = pd.DataFrame({
    "Feature": data_features1.columns,
    "Importance": fp.importances_mean
}).sort_values(by='Importance', ascending=False)

# Barplot of feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Feature", y="Importance", data=fp_importances,yerr=fp.importances_std, capsize=.2)
plt.title("Feature Permutation Importance Barplot With Error Bar")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=-30)
plt.show()

# Sort by importance and select the top 5 features
top_features = fp_importances.head(5)
selected_features = top_features['Feature'].tolist()
print("Selected Top 5 Features:", selected_features)

# Calculate accuracy on validation set
fea_val_selected = fea_val[selected_features]
fp_val = RandomForestClassifier(random_state=3,n_estimators=300,max_depth=20)
fp_val.fit(fea_train[selected_features], tar_train)
tar_val_pred = fp_val.predict(fea_val_selected)
val_accuracy = accuracy_score(tar_val, tar_val_pred)

print(f"Feature Permutation Validation accuracy with top 5 features: {val_accuracy:.2f}")

# KFold cross-validator
from sklearn.model_selection import KFold, cross_val_score

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=2)
scores = cross_val_score(fp_val, data_features1, data_targets1, cv=kf, scoring='accuracy')
print("k-fold cross valuation:")
print("Scores for each fold:", scores)
print("Mean accuracy:", scores.mean())




# Leaning curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    rf,
    fea_train,
    tar_train,
    cv=8,  # number of folds
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

learning_curve_data = pd.DataFrame({
    'Training Size': train_sizes,
    'Train Score': train_scores_mean,
    'Test Score': test_scores_mean,
    'Train Std': train_scores_std,
    'Test Std': test_scores_std
})

# Use Seaborn to draw the learning curve
plt.figure(figsize=(10, 6))
sns.lineplot(data=learning_curve_data, x='Training Size', y='Train Score', label='Training Score', marker='o')
sns.lineplot(data=learning_curve_data, x='Training Size', y='Test Score', label='Validation Score', marker='o')

# Fill the standard deviation area
plt.fill_between(learning_curve_data['Training Size'], 
                 learning_curve_data['Train Score'] - learning_curve_data['Train Std'],
                 learning_curve_data['Train Score'] + learning_curve_data['Train Std'], 
                 color='blue', alpha=0.2)
plt.fill_between(learning_curve_data['Training Size'], 
                 learning_curve_data['Test Score'] - learning_curve_data['Test Std'],
                 learning_curve_data['Test Score'] + learning_curve_data['Test Std'], 
                 color='orange', alpha=0.2)

plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Accuracy Score')
plt.legend()
plt.grid()
plt.show()


