import tensorflow as tf
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import pandas as pd
df=pd.read_csv('train.csv')
array=df.values
#print(df.head(50))
x=array[0:,0:20]
y=array[0:,20]
validation_size=0.35
seed=42
x_train,x_validation,y_train,y_validation=model_selection.train_test_split(x,y,test_size=validation_size,random_state=seed)


lr=DecisionTreeClassifier()
lr.fit(x_train,y_train)


predictions=lr.predict(x_validation)
print(accuracy_score(y_validation,predictions))
