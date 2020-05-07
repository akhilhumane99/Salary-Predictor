# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

'''from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
sc_Y=StandardScaler()
Y_train=sc_y.fit_transform(Y_train)'''

#Fitting dataset to the regressor 
from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)

#Visualizing Decision Tree Regression in smoother way and in higher resolution
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Truth or bluff (Decision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


#Predicting a new result
Y_pred=regressor.predict(np.array([[6.5]]))