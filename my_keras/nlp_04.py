# 循环神经网络训练情感分析
# 使用长短记忆模型LSTM处理情感分类
# LSTM是循环神经网络的一种，本质上，它按照时间顺序，把信息进行有效的整合和筛选
# 有的信息得到保留，有的信息被丢弃，在时间t时刻，你获得的信息比如对段落的理解
# 会包含之前的信息，之前提到的事件、人物等
# LSTM说，根据我手里的训练数据，我得找出一个方法来如何进行有效得信息取舍，从而把最有价值
# 的信息保留到最后，那么最自然的想法是总结出一个规律用来处理前一时刻，并和当前t时刻的信息
# 进行融合
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.layers import Dense
from keras.preprocessing import sequence
from keras.datasets import imdb
import numpy as np

(X_train, y_train), (X_test, y_test) = imdb.load_data()
max_word = 400
X_train = sequence.pad_sequences(X_train, maxlen=max_word)
X_test = sequence.pad_sequences(X_test, maxlen=max_word)
vocab_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])]) + 1

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_word))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=100)
scores = model.evaluate(X_test, y_test)
print(scores)
# 预测的精确度大致为86.7%
