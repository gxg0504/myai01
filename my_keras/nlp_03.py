# 卷积神经网络训练情感分析
# 全连接神经网络几乎对网络模型没有任何限制，缺点过拟合，参数多，实际应用中，对模型加
# 一定的限制，参数会大幅减少，降低了模型的复杂度，模型的普适性进而会提高

# 在自然语言处理领域，卷积的作用在于利用文字的局部特征。一个词的前后几个词比如和这个词
# 本身相关，这组成该词所代表的词群，词群进而会对段落文字的意思进行影响，决定这个段落到底
# 是正向的还是负向的
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import numpy as np
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data()
max_word = 400
X_train = sequence.pad_sequences(X_train, maxlen=max_word)
X_test = sequence.pad_sequences(X_test, maxlen=max_word)
vocab_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])]) + 1

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_word))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=100)
scores = model.evaluate(X_test, y_test, verbose=1)
print(scores)
# 精确度提高了一点，在85.5%，可以试着调整模型的参数，增加训练次数








