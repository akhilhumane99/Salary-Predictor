# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

'''from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)'''

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X=sc_X.fit_transform(np.array([X]).reshape(-1, 1))
Y=sc_Y.fit_transform(np.array([Y]).reshape(-1, 1))

#Fitting SVR to the dataset
#Create SVR model
from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf')
regressor.fit(X,Y)

#Visulaizing polynomial results
plt.scatter(X,Y,color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title("Truth or bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


######Predicting a new result

Y_pred=sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#############only for single sample ,recompile the program after deleting this line...

#Y_pred= regressor.predict(np.array([6.5]).reshape(1, 1))



#Visualizing in smoother way
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title("Truth or bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


