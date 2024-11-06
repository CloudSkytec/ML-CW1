import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier 


wine_quality = fetch_ucirepo(id=186) 
df = pd.concat([wine_quality.data.features , wine_quality.data.targets], axis=1)

# check dataset
df.head()
df.describe()
df.info()

## combine
df = df.drop(columns=['density','free_sulfur_dioxide'])

# todo outliers treatment
df['alcohol'] = winsorize(df['alcohol'], limits=[0.001, 0.001])
df['residual_sugar'] = winsorize(df['residual_sugar'], limits=[0.001, 0.001])
df['sulphates'] = winsorize(df['sulphates'], limits=[0.001, 0.003])



# Apply log transformation (add a small constant to avoid log(0))
df['chlorides'] = np.log(df['chlorides'] + 1e-5)
df['sulphates'] = np.log(df['sulphates'] + 1e-5)
df['residual_sugar'] = np.log(df['residual_sugar'] + 1e-5)
df['total_sulfur_dioxide'] = np.log(df['total_sulfur_dioxide'] + 1e-5)

# remove less important features
df = df.drop(columns=['citric_acid'])

# smote
oversample = SMOTE(random_state=42,k_neighbors=4)
# transform the dataset
X, y = oversample.fit_resample(df.drop(columns=['quality']), df['quality'])
ys = y.values.ravel()
df_resampled = pd.concat([X,y], axis=1)

# spilt
X_train, X_test, y_train, y_test = train_test_split(X, ys, test_size=0.2, random_state=42)

# random forest
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)
model = DecisionTreeClassifier(min_samples_split=3, random_state=42)
model.fit(X_train, y_train)

# logistic regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1, solver='saga', max_iter=500, penalty='l1').fit(X_train,y_train)
print(clf.score(X_test,y_test))


# KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)
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