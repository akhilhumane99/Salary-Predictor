import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
dataset= pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
#cross validation

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#fitting Simple linear regression to training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#predicting

Y_pred = regressor.predict(X_test) 

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor(X_train),color='blue')
plt.title('Salary vs Experience (Training sets)')
plt.xlabel("Years of Experience" )
plt.ylabel("Salary")