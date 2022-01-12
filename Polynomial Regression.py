#Polynomial Regression
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

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)# X will be converted to X_poly which consists of b0+ b1X1 + b2X1^2 + .... till +b4X1^4 which is the degree = 4
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualizing the Linear Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color ='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color ='blue')
#Or plt.plot(X, lin_reg_2.predict(X_poly), color ='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X)+0.1, 0.1) # to visualze in a better way: instead of specifice lines with "1" unit difference we use this to appeare more smoothily
#max(X)+0.1 to reach the last point
X_grid = X_grid.reshape(len(X_grid),1) #reshape to convert it to matrix
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color ='blue')
#Or plt.plot(X, lin_reg_2.predict(X_poly), color ='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(np.array([6.5]).reshape(1, 1)) #reshape to array(1,1)to avoid error of expected 2D array

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1, 1)))