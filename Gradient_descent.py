import numpy as np

#类的目的是求解斜率和截距
class Linear_model(object):
    def __init__(self):
        self.w = np.random.randn(1)[0]

        self.b = np.random.randn(1)[0]
        print('初始值:\n')
        print(self.w,self.b)

    #model函数的返回值是假设函数H（x）= X*w + b
    def model(self,x):
        return self.w * x + self.b

    #计算梯度
    def loss(self,x,y):
        cost = (y-self.model(x))**2

        g_w = 2*(y-self.model(x))*(-x)
        g_b = 2*(y-self.model(x))*(-1)

        return g_w,g_b

    #更新斜率和截距
    def gradient_descend(self,g_w,g_b,step=0.01):

        self.w = self.w - step*g_w
        self.b = self.b - step*g_b
        print('更新后的值:\n')
        print(self.w,self.b)

    def  fit(self,X,y):

        w_last = self.w + 1
        b_last = self.b + 1

        precision = 0.00001
        max_count = 3000
        count = 0
        while True:
            if (np.abs(self.w - w_last) < precision) and (np.abs(self.b - b_last) < precision):
                break

            w_last = self.w
            b_last = self.b
            if count > max_count:
                break

            g_w = 0
            g_b = 0
            size = X.shape[0]
            for i,j in zip(X,y):
                g_w += self.loss(i,j)[0]
                g_b += self.loss(i,j)[1]
            g_w = g_w / size
            g_b = g_b / size
            self.gradient_descend(g_w,g_b)

            count += 1


    def coef_(self):
        return self.w

    def intercept_(self):
        return self.b


X = np.linspace(2.5,12,25)
w = np.random.randint(2,10,size=1)[0]
b = np.random.randint(-5,5,size=1)[0]
print(w)
print(b)
y = X*w + b + np.random.randn(25)*2

lm = Linear_model()
lm.fit(X,y)



