#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset

dataset=pd.read_csv('Data.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values
print(x[:,0])
#taking care of missing data

#from sklearn.impute import SimpleImputer 

#from sklearn.preprocessing import Imputer

#imputer=SimpleImputer(missing_values='NaN' ,strategy='mean')
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer=imputer.fit(x[:,1:3])

x[:,1:3]=imputer.transform(x[:,1:3])


#encoding categorical data 

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
onehotencoder=OneHotEncoder(categorical_features = [0])
x=onehotencoder.fit_transform(x).toarray()

y=labelencoder_x.fit_transform(y)

#spitting the dataset into the training and testing set
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0) 

#feature scaling 
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)



