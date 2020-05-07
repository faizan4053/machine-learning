import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(train_x,train_y)

y_pred=lr.predict(test_x)

plt.scatter(train_x,train_y,color='blue')

plt.plot(train_x,lr.predict(train_x),color='red')
plt.title('Salary VS Experience(Training Data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(test_x,test_y,color='blue')

plt.plot(train_x,lr.predict(train_x),color='red')
plt.title('Salary VS Experience(Test Data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()