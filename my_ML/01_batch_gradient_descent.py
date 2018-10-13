#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: batch_gradient_descent.py

import numpy as np
__author__ = 'yasaka'
#固定随机种子
np.random.seed(1)
#创建模拟训练集
X = 2 * np.random.rand(10000, 1)
y = 4 + 3 * X + np.random.randn(10000, 1)
X_b = np.c_[np.ones((10000, 1)), X]
# print(X_b)

learning_rate = 0.1
n_iterations = 500
#有100条样本
m = 10000

#初始化θ
theta = np.random.randn(2, 1)
count = 0


for iteration in range(n_iterations):
    count += 1
    gradients = 1/m * X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - learning_rate * gradients

print(count)
print(theta)









