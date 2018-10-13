#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: stochastic_gradient_descent.py

import numpy as np
from sklearn.linear_model import SGDRegressor

__author__ = 'yasaka'
#固定随机种子
np.random.seed(1)
#生成训练集
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
print(X_b)

n_epochs = 1000
t0, t1 = 5, 50

m = 100

#设置可变学习率
def learning_schedule(t):
    return t0 / (t + t1)

#初始化θ
theta = np.random.randn(2, 1)

learning_rate = 0.1
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        # 为了解决维度问题，在索引屈指
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta)-yi)
        learning_rate = learning_schedule(epoch*m + i)
        theta = theta - learning_rate * gradients

print(theta)

SGDRegressor




