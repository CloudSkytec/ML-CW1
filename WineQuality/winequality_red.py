import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
