# Random Forest Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[ : , 1:2].values
y = dataset.iloc[ : , 2].values

"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0, shuffle = False, stratify = None)
"""

# In Simple Linear Regression Library we don't need feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor #ensemble in general means you have hybrid of multiple algorithms together and you take the average prediction or hybrid of the same algorithm like multiple decision trees
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0) #n_estimators: s the number of required trees, criteron: the way to calculate the average "mse" or median or ....etc
regressor.fit(X,y)

# Predicting a new result with Random Forest Regression
y_pred = regressor.predict(np.array([6.5]).reshape(1, 1)) #reshape to array(1,1)to avoid error of expected 2D array

#Visualizing the Random Forest Regression Results with high resolution
X_grid = np.arange(min(X), max(X), 0.1) # to visualze in a better way: instead of specifice lines with "1" unit difference we use this to appeare more smoothily
#max(X)+0.1 to reach the last point
X_grid = X_grid.reshape(len(X_grid),1) #reshape to convert it to matrix
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color ='blue')
#Or plt.plot(X, lin_reg_2.predict(X_poly), color ='blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()