#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: linear_regression_1.py

import numpy as np
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression

__author__ = 'yasaka'
#生成模拟数据集
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
lin_reg = LinearRegression()
#通过fit训练模型，训练完成后，参数会保存在对象中
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
X_new = np.array([[0], [2]])
#predict
print(lin_reg.predict(X_new))








