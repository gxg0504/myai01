# 情感分析
# 通过一个电影评论的例子详细讲解深度学习在情感分析中的关键技术

import numpy as np
from keras.datasets import imdb
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = imdb.load_data()
print(X_train[0])
print(len(X_train[0]))
print(X_train[1])
print(len(X_train[1]))

# 原来，Keras自带的load_data函数帮我们从亚马逊S3中下载了数据，并且给每个词标注了
# 一个索引（index），创建了字典。每段文字的每个词对应了一个数字。

# 1表示正面，0表示负面，(25000,)
print(X_train.shape)
print(y_train.shape)
print(y_train[:10])
"""
# 接下来看看平均每个评论有多少个字
avg_len = list(map(len, X_train))
print(np.mean(avg_len))
# 为了直观显示，这里画一个分布图
plt.hist(avg_len, bins=range(min(avg_len), max(avg_len) + 50, 50))
plt.show()
# 第一，文字分词。英语分词可以按照空格分词，中文分词可以参考jieba
# 第二，建立字典，给每个词标号
# 第三，把段落按字典翻译成数字，变成一个array
# 接下来就可以开始建模了
"""


































