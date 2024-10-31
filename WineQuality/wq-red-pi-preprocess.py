import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split

# read the CSV file
df = pd.read_csv('winequality-red.csv', delimiter=';')

# check dataset
df.head()
df.describe()
df.info()

## combine
df = df.drop(columns=['density','free sulfur dioxide'])

# todo outliers treatment
df['alcohol'] = winsorize(df['alcohol'], limits=[0.005, 0.005])
df['residual sugar'] = winsorize(df['residual sugar'], limits=[0.005, 0.005])

# Apply log transformation (add a small constant to avoid log(0))
df['chlorides'] = np.log(df['chlorides'] + 1e-5)
df['sulphates'] = np.log(df['sulphates'] + 1e-5)
df['residual sugar'] = np.log(df['residual sugar'] + 1e-5)
df['total sulfur dioxide'] = np.log(df['total sulfur dioxide'] + 1e-5)

# remove less important features
df = df.drop(columns=['citric acid'])

# smote
from imblearn.over_sampling import SMOTE #conda install conda-forge::tpot-imblearn
oversample = SMOTE(k_neighbors=4)
# transform the dataset
X_toresample = df.drop(columns=['quality'])
y_toresample = df['quality']

X_resampled, y_resampled = oversample.fit_resample(X_toresample, y_toresample)
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
X = df_resampled.drop(columns=['quality'])
y = df_resampled['quality']

# Selecting features for scaling, excluding the target 'quality'
min_max_scaler = MinMaxScaler()
features_to_scale = df_resampled.columns.drop('quality')
df_resampled[features_to_scale] = min_max_scaler.fit_transform(df_resampled[features_to_scale])

ys = y.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, ys, test_size=0.2, random_state=42)

model = RandomForestClassifier(min_samples_leaf=1, min_samples_split=5, n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# KFold cross-validator
kf = KFold(n_splits=8, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("k-fold cross valuation:")
print("Scores for each fold:", scores)
print("Mean accuracy:", scores.mean())

# box plots
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df_resampled.items():
    if col != 'type':
        sns.boxplot(y=col, data=df_resampled, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df_resampled.items():
    if col != 'type':
        sns.histplot(value, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)