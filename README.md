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
  - The data description suggests that an anomaly detection algorithm can be used to identify a small number of good or bad wines. Also, it is not possible to determine whether all input variables are related. Therefore, special attention needs to be paid to the   processing of data.

- **Regression Dataset**:  [Abalone - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/1/abalone)
  - Predict the age of abalone from physical measurements
  - The data description has indicated that the default values have been removed and the data scaling is reasonable. Therefore, it is more convenient in the display and use of data, and it is easy to apply the verification of the model.

# 2. Method Description
- **Model**
  - ***Linear regression***:
  - ***Logistic regression***:
  - ***Support vector machines (SVM)***:
  - ***Decision trees***:
  - ***Multilayer perceptron neural network***:
- **Evaluate && Compare**
  - ***K-fold cross-validation***:
  - ***Mean squared error (MSE) for Regression***:

## 2.1 Analyzing Data

### 2.1.1  Wine Quality






### 2.1.2 Abalone





## 2.2 Data Preprocessing

### 2.2.1 Wine Quality

todo: given by above analyzations, listing preprocessing methods, e.g.(data normalization, feature selections)  

### 2.2.2 Abalone



# 3. Parameters settings



# 4. Assessment

#### Logistic Regression Model 

After performing proper data normalization and selecting optimal hyperparameters, we found that the performance remains difficult to compare with three other models.

There are several reasons for this. Based on the principles of Logistic Regression, features should be linearly related to the target value. However, by analyzing the correlation of each feature with quality, it is evident that customers tend to choose specific ranges of parameters such as 'alcohol', 'residual sugar', 'total SO2', and so on. It turns out that the combination of specific range of wine parameters is crucial.


# 5. Conclusion
