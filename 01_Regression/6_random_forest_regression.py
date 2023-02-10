from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Random Forest Regression model on the whole dataset
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])

# Evaluate Simple Linear Regression model by MAPE , R score


def mape(y_true, y_pred):
    Accuracy = r2_score(y_true, y_pred)*100
    Mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(" Accuracy of the model is %.2f" % Accuracy)
    print(" MAPE of the model is %.2f" % Mape)


mape(y, regressor.predict(X))


# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
