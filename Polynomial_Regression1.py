# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,Y)


#visualizing the linear regression results

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.title("Truth or bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


#Visulaizing polynomial results
#X_grid = np.arange(min(X),max(X),0.1)
#X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color="blue")
plt.title("Truth or bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Predicting Output
lin_reg.predict(np.array([[6.5]]))
 
lin_reg_2.predict(poly_reg.fit_transform(np.array([[6.5]])))
