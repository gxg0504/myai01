from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
print(iris.keys())
print(type(iris))
print(iris['DESCR'])
print(iris['target_names'])
print(iris['feature_names'])
print(iris['target'].shape)
print(iris['data'].shape)
X = iris['data']
y = iris['target']
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.85,random_state=42)
print(X_train.shape,y_test.shape)
lr = LogisticRegression(C=100,multi_class='multinomial',solver='lbfgs')
lr.fit(X_train,y_train)
y_hat = lr.predict(X_test)
y_prob=lr.predict_proba(X_test)
#numpy的技巧
print('prob:',y_prob)
print(sum(y_prob[0]))
print(sum(y_prob[1]))
print('y_hat',y_hat)
print()
print(accuracy_score(y_test,y_hat))
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_hat= rf.predict(X_test)
print(accuracy_score(y_test,y_hat))
DecisionTreeClassifier
