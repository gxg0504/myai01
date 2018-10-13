#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: linear_regression_0.py

import numpy as np
import matplotlib.pyplot as plt
__author__ = 'yasaka'

seed = np.random.seed(123)
# rand 均匀分布 给出一个0-1的均匀分布的随机数
n_samples= 10000
X = 2 * np.random.rand(n_samples, 1)
# print(X)
y = 4 + 3 * X + np.random.randn(n_samples, 1)
# print(y)
X_b = np.c_[np.ones((n_samples, 1)), X]
print(X_b.shape)
print(X_b)
#linear algraba 线性代数
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)
X_new = np.array([[0], [2]])
X_new_b = np.c_[(np.ones((2, 1))), X_new]
print(X_new_b)
y_predict = X_new_b.dot(theta_best)
print(y_predict)
plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()



