# 多层全连接神经网络训练情感分析
# Keras提供了设计嵌入层的模板，只要在建模的时候加一行Embedding Layer函数的代码就可以
# 注意，嵌入层一般是需要通过数据学习的，也可以借用已经训练好的嵌入层比如Word2Vec
# 中预训练好的词向量直接放入模型，或者把预训练好的词向量作为嵌入层初始值，再进行训练
# Embedding函数定义了嵌入层的框架，其一般有3个变量：字典的长度（即文本中有多少词向量）
# 词向量的维度和每个文本输入的长度
# 每个文本或长或短，所以可以采用Padding技术取最长的文本长度作为文本的输入长度，不足长度的
# 都用空格填满，即把空格当成一个特殊字符处理，空格本身一般也会被赋予词向量，这可以通过
# 机器学习训练出来。
# Keras提供了sequence.pad_sequences函数帮我们做文本的处理和填充工作
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import numpy as np
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data()
# 使用下面的命令计算最长的文本长度：
m = max(max(list(map(len, X_train))), max(list(map(len, X_test))))
print(m)
# 从中我们会发现有一个文本特别长，居然有2494个字符，这种异常值需要排除，考虑到文本
# 的平均长度为230个字符，可以设定最多输入的文本长度为400个字符，不足400个字符的文本用
# 空格填充，超过400个字符的文本截取400个字符
max_word = 400
X_train = sequence.pad_sequences(X_train, maxlen=max_word)
X_test = sequence.pad_sequences(X_test, maxlen=max_word)
vocab_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])]) + 1
print(vocab_size)
# 这里1代表空格，其索引被认为是0
# 下面从最简单的多层神经网络开始尝试
# 首先建立序列模型，逐步往上搭建网络
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_word))
# 第一层是嵌入层，定义了嵌入层的矩阵为vocab_size*64，每个训练段落为其中的max_word*64
# 矩阵，作为数据的输入，填入输入层
model.add(Flatten())
# 把输入层压平，原来是max_word*64的矩阵，现在变成一维的长度为max_word*64的向量
# 接下来不断搭建全连接神经网络，使用relu函数，最后一层是Sigmoid，预测0，1变量的概率
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20
          , batch_size=50, verbose=1)
score = model.evaluate(X_test, y_test)
print(score)
# 其精确度大约在85%，如果多做几次迭代，精确度会更高
