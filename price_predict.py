from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets

#波士顿房价，跟好多因素有关系
data = datasets.load_boston()
X = data['data']
y = data['target']
#数据集划分
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)


#模型训练
lr = LinearRegression(fit_intercept=False)
lr.fit(X_train,y_train)

#算法预测
print(lr.predict(X_test).round(2))

#w_表示每个属性特征值所占权重，b_表示偏差
w_ = lr.coef_
b_ = lr.intercept_

