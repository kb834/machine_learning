'''
    用线性回归算法预测糖尿病
    并且使用R^2分数决定系数法，均方误差...等评估指标评估线性回归算法预测结果
'''

import sklearn.datasets  as dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

diabetes = dataset.load_diabetes()

X = diabetes['data']
y = diabetes['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

lr = LinearRegression()
lr.fit(X_train,y_train)
result = lr.predict(X_test).round(2)

#线性回归的评价指标：决定系数
score1 = lr.score(X_test,y_test)

#R^2分数，决定系数回归指标的公式
u = ((y_test-result)**2).sum()
v = ((y_test-y_test.mean())**2).sum()
score2 = 1 - u/v


#R^2分数，决定系数，系数越大，预测越准
print(r2_score(y_test,result))
#平均绝对误差，误差越小，预测越准
print(mean_absolute_error(y_test,result))
#均方误差，误差越小，预测越准
print(mean_squared_error(y_test,result))
#均方对数误差，误差越小，预测越准
print(mean_squared_log_error(y_test,result))
