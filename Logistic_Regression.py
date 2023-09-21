from sklearn.linear_model import LogisticRegression
import sklearn.datasets as dataset
from sklearn.model_selection import train_test_split

iris = dataset.load_iris()
#iris中有150个样本，4个属性，分别包括花萼长宽，花瓣的长宽
X = iris['data']
y = iris['target']

X_train,X_test,y_train,y_test = train_test_split(X,y)

lg = LogisticRegression()
lg.fit(X_train,y_train)
result = lg.predict(X_test)
print(result)
print(y_test)
print(lg.score(X_test,y_test))
