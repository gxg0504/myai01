import numpy as np
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

# 读取数据
all_text = open("四世同堂.txt", encoding='utf-8').read()
print(all_text[10])

# 1, 首先对所有待建模的单字和字符进行索引
# 用set函数抽取的每个单字的集合按照编码从小到大排序
sorted_charset = sorted(set(all_text))
print(sorted_charset)
# 对一个单字进行编号索引
char_indices = dict((c, i) for i, c in enumerate(sorted_charset))
# 对每个索引建立单字的词典，主要是为了方便预测出来的索引标号向量转换为人能够阅读的文字
indice_char = dict((i, c) for i, c in enumerate(sorted_charset))

# 2, 其次构造句子序列
max_len = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(all_text) - max_len, step):
    sentences.append(all_text[i: i + max_len])
    next_chars.append(all_text[i + max_len])
print('nb sequences:', len(sentences))

# 3, 然后建立神经网络模型，对索引标号序列进行向量嵌入后的向量构造长短记忆神经网络
print('Vectorization...')
X = np.zeros((len(sentences), max_len, len(sorted_charset)), dtype=np.bool)
y = np.zeros((len(sentences), len(sorted_charset)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    if i % 30000 == 0:
        print(i)
        for t in range(max_len):
            char = sentence[t]
            X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


def data_generator(X, y, batch_size):
    if batch_size < 1:
        batch_size = 256
    number_of_batches = X.shape[0]  # batch_size
    counter = 0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    # reset generator
    while 1:
        index_batch = shuffle_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = (X[index_batch, :, :]).astype('float32')
        y_batch = (y[index_batch, :]).astype('float32')
        counter += 1
        yield (np.array(X_batch), y_batch)
        if counter < number_of_batches:
            np.random.shuffle(shuffle_index)
            counter = 0


# 构建模型，一个LSTM
batch_size = 256
print('Build Model...')
model = Sequential()
model.add(LSTM(256, batch_size=batch_size, input_shape=(max_len,
                                                        len(sorted_charset)),
               recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(len(sorted_charset)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit_generator(data_generator(X, y, batch_size), steps_per_epoch=X.shape[0],
                    epochs=50)

# 4, 最后我们来检验建模效果
start_index = 2799
sentence = all_text[start_index: start_index + max_len]
sentence0 = sentence
x = np.zeros((1, max_len, len(sorted_charset)))
for t, char in enumerate(sentence):
    x[0, t, char_indices[char]] = 1.


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    scaled_preds = preds ** (1/temperature)
    preds = scaled_preds / np.sum(scaled_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# 接下来一次对每一句的下20个字符进行预测，并根据预测得到的索引标号找出对应的文字供人阅读
generated = ''
ntimes = 20
for i in range(ntimes):
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, 0.1)
    next_char = indice_char[next_index]
    generated += next_char
    sentence = sentence[1:]+next_char
    print(sentence)


