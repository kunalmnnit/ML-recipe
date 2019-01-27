import sklearn
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

from sklearn import model_selection
X_train ,X_test ,y_train ,y_test = model_selection.train_test_split(X, y, test_size=0.2)

from sklearn import tree
clf = tree.DecisionTreeClassifier()

clf.fit(X_train,y_train)

predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))
