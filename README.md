# ML-CW1 

**Group Number:** C4_18

**Group Members:** 

- Juntian Xiao
- Luqi Xin
- Guangzheng Dong
- Yuhong Yuan
- Qifeng He

# 1. Introduction

- **Classification Dataset** : [Wine Quality - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality) 
  - Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests (see [Cortez et al., 2009], http://www3.dsi.uminho.pt/pcortez/wine/).

- **Regression Dataset**:  [Abalone - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/1/abalone)
  - Predict the age of abalone from physical measurements

# 2. Method Description

## 2.1 Analyzing Data

### 2.1.1  Wine Quality



### 2.1.2 Abalone





## 2.2 Data Preprocessing

### 2.2.1 Wine Quality

#### Scaling

Given that features like `fixed acidity`, `volatile acidity`, `residual sugar`, and `alcohol` have different ranges, standard scaling can help normalize these variations.

#### Outliers

By analyzing the graph of scaled features, we applied the `winsorize()` function to trim some extreme values, making the scaled data balanced.



### 2.2.2 Abalone



# 3. Parameters settings
## 3.1 Wine Quality
#### 3.1.1 Random Forest Classifier
**parameters**:  **min_samples_split**=5, **n_estimators**=200, **random_state**=42

- **min_sample_split:** By increasing this parameter, we have reduced the total number of splits, thereby limiting the number of parameters in the model and reducing overfitting.

  **n_estimators:** Given the limited size of our dataset, increasing this parameter could result in a more generalized model.




# 4. Assessment

#### Logistic Regression Model 

After performing proper data normalization and selecting optimal hyperparameters, we found that the performance remains difficult to compare with three other models.

There are several reasons for this. Based on the principles of Logistic Regression, features should be linearly related to the target value. However, by analyzing the correlation of each feature with quality, it is evident that customers tend to choose specific ranges of parameters such as 'alcohol', 'residual sugar', 'total SO2', and so on. It turns out that the combination of specific range of wine parameters is crucial.


# 5. Conclusion