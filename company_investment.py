import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read dataset
dataset=pd.read_csv("50_Startups.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#encoding categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le=LabelEncoder()
x[:,3]=le.fit_transform(x[:,3]);

ohe=OneHotEncoder(categorical_features=[3])

x=ohe.fit_transform(x).toarray()

#avoiding dummy variable trap
x=x[:,1:]

#traing and testing

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_train,y_train);

#predicting the test set results

y_predict=lr.predict(x_test)

#preparing for backword elimination
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)

import statsmodels.api as sm

x_optimal=x[:,[0,1,2,3,4,5]]

regessor_OLS=sm.OLS(endog=y,exog=x_optimal).fit()
regessor_OLS.summary()
x_optimal=x[:,[0,1,3,4,5]]

regessor_OLS=sm.OLS(endog=y,exog=x_optimal).fit()
regessor_OLS.summary()
x_optimal=x[:,[0,3,4,5]]

regessor_OLS=sm.OLS(endog=y,exog=x_optimal).fit()
regessor_OLS.summary()

x_optimal=x[:,[0,3,5]]

regessor_OLS=sm.OLS(endog=y,exog=x_optimal).fit()
regessor_OLS.summary()

y_pred=regessor_OLS.predict(x_test)


