# Machine Learning With Python

> 自學 Python 、Machine Learning，並紀錄程式碼筆記

以 Python 實現機器學習。資料篩選、建構模型、預測結果、模型驗證、視覺化。

參考課程：[Machine Learning A-Z™: Python & R in Data Science [2023]](https://www.udemy.com/course/machinelearning/)

- **Regression 迴歸**
  - [簡單線性迴歸](#簡單線性迴歸Simple-Linear-Regression)
  - [複迴歸](#複迴歸Multiple-Linear-Regression)
  - [多項式迴歸](#多項式迴歸Polynomial-Regression)
  - [支持向量迴歸](#支持向量迴歸SVR)
  - [決策樹迴歸](#決策樹迴歸Decision-Tree-Regression)
  - [隨機森林迴歸](#隨機森林迴歸Random-Forest-Regression)
- **Classification 分類**
  - [羅吉斯迴歸](#羅吉斯迴歸Logistic-Regression)
  - [K-近鄰演算法](#K-近鄰演算法KNN)
  - [支援向量機](#支援向量機SVM)
  - [非線性支援向量機](#非線性支援向量機Kernel-SVM)
  - [樸素貝氏分類器](#樸素貝氏分類器Native-Bayes)
  - [決策樹](#決策樹Decision-Tree)
  - [隨機森林](#隨機森林Random-Forest)
- **Clurtering 分群**

  - [K-平均演算法](#K-平均演算法K-means-Clustering)
  - [階層式分群法](#階層式分群法Hierarchical-Clustering)

- [模型驗證](#模型驗證)
  - [迴歸模型指標](#迴歸模型指標)
  - [分類模型指標](#分類模型指標)

## 資料前處理

- #### 資料輸入

```python
import pandas as pd

# 輸入資料
dataset = pd.read_csv('data/Salary_Data.csv')

# 定義反應變數、解釋變數
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
```

- #### 類別資料轉換

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
```

- #### 資料切割

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0)
```

## Regression

### 簡單線性迴歸（Simple Linear Regression）

- 建立模型

```python
from sklearn.linear_model import LinearRegression

# 訓練模型
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# 預測結果
y_pred = regressor.predict(x_test)
```

- 模型驗證
  參照 [迴歸模型指標](#迴歸模型指標)。

![image](./01_Regression/image/simple%20linear%20regression%20traing%20set.png)

### 複迴歸（Multiple Linear Regression）

- 建立模型

```python
from sklearn.linear_model import LinearRegression

# 訓練模型
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# 預測結果
y_pred = regressor.predict(x_test)
```

- 模型驗證
  參照 [迴歸模型指標](#迴歸模型指標)。

![image](./01_Regression/image/multiple%20linear%20regression.png)

### 多項式迴歸（Polynomial Regression）

- 建立模型

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 訓練模型
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# 預測結果
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
```

- 模型驗證

```python
from sklearn.metrics import r2_score

# 以 R squared、MAPE 驗證模型
def mape(y_true, y_pred):
    Accuracy = r2_score(y_true, y_pred)*100
    Mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(" Accuracy of the model is %.2f" % Accuracy)
    print(" MAPE of the model is %.2f" % Mape)


mape(Y, lin_reg_2.predict(poly_reg.fit_transform(X)))

```

![image](./01_Regression/image/polynomial%20line%20smooth.png)

### 支持向量迴歸（SVR）

### 決策樹迴歸（Decision Tree Regression）

### 隨機森林迴歸（Random Forest Regression）

### 羅吉斯迴歸（Logistic Regression）

### K-近鄰演算法（KNN）

### 支援向量機（SVM）

### 非線性支援向量機（Kernel SVM）

### 樸素貝氏分類器（Native Bayes）

### 決策樹（Decision Tree）

### 隨機森林（Random Forest）

## Classification

### 羅吉斯迴歸（Logistic Regression）

### K-近鄰演算法（KNN）

### 支援向量機（SVM）

### 非線性支援向量機（Kernel SVM）

### 樸素貝氏分類器（Native Bayes）

### 決策樹（Decision Tree）

### 隨機森林（Random Forest）

## Clurtering

### K-平均演算法（K means Clustering）

### 階層式分群法（Hierarchical Clustering）

## 模型驗證

- ### 迴歸模型指標

```python
from sklearn.metrics import r2_score

# 以 R squared、MAPE 驗證模型
def mape(y_true, y_pred):
    Accuracy = r2_score(y_true, y_pred)*100
    Mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(" Accuracy of the model is %.2f" % Accuracy)
    print(" MAPE of the model is %.2f" % Mape)


mape(y_test, y_pred)

```

- ### 分類模型指標
