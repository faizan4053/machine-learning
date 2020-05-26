#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#getting the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t',quoting=3)

#cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
cleaned_dataset=[]
for i in range(0,1000):
    #remove all unnecessary characters like , ,. ,..., etc from words
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    
    #removing unnecessary words like this,that or the,and 
    #performing stemming ,i.e, converting words to present or absolute form
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    #converting back to string from list
    review=' '.join(review)
    cleaned_dataset.append(review)

#creating a bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500) #max_features will get the top 1500 comman features
x=cv.fit_transform(cleaned_dataset).toarray() #independent variable(sparse matrix)
y=dataset.iloc[:,1].values  #dependent variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.40, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('negative reviews:')
print(cm[0][0])
print('positive reviews:')
print(cm[1][1])
accuracy=(cm[0][0]+cm[1][1])/(0.4*(1000))
print(accuracy*100)