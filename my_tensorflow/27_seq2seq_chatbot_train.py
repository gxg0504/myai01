import tensorflow as tf
import numpy as np

# run 'data.py' to produce some files we needed.
# run 'train.py' to train the model.
# run 'test_model.py' to predict.

batch_size = 27
sequence_length = 20
hidden_size = 256
num_layers = 2
num_encoder_symbols = 1004  # 'UNK' and '<go>' and '<eos>' and '<pad>'
num_decoder_symbols = 1004
embedding_size = 256
learning_rate = 0.001
model_dir = './model'

# 创建四个占位符，分别为encoder的输入占位符、decoder的输入占位符和decoder的target占位符，
# 还有权重占位符。其中batch_size是输入样本一批的数量，sequence_length为我们定义的序列的长度。

encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])

targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
weights = tf.placeholder(dtype=tf.float32, shape=[batch_size, sequence_length])

# 创建循环神经网络结构，这里使用LSTM结构，hidden_size是隐含层数量，用MultiRNNCell是
# 因为我们希望创建一个更复杂的网络，num_layers为LSTM的层数。
cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)


def loadQA():
    train_x = np.load('./idx_q.npy', mmap_mode='r')
    train_y = np.load('./idx_a.npy', mmap_mode='r')
    train_target = np.load('./idx_o.npy', mmap_mode='r')
    return train_x, train_y, train_target


# 使用TensorFlow为我们准备好了的embedding_rnn_seq2seq函数搭建seq2seq结构，
# 当然我们也可以自己从LSTM搭起，分别创建encoder和decoder，但为了方便直接使用
# embedding_rnn_seq2seq即可。使用tf.unstack函数是为了将encoder_inputs和
# decoder_inputs展开成一个列表，num_encoder_symbols和num_decoder_symbols
# 对应到我们的词汇数量。embedding_size则是我们的嵌入层的数量，feed_previous这个变量很重要
# ，设为False表示这是训练阶段，训练阶段会使用decoder_inputs作为decoder的其中一个输入，
# 但feed_previous为True时则表示预测阶段，而预测阶段没有decoder_inputs，所以只能依靠
# decoder上一时刻输出作为当前时刻的输入。
results, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    tf.unstack(encoder_inputs, axis=1),
    tf.unstack(decoder_inputs, axis=1),
    cell,
    num_encoder_symbols,
    num_decoder_symbols,
    embedding_size,
    feed_previous=False
)
logits = tf.stack(results, axis=1)
print("sssss: ", logits)

# 接着使用sequence_loss来创建损失，这里根据embedding_rnn_seq2seq的输出来计算损失，
# 同时该输出也可以用来做预测，最大的值对应的索引即为词汇的单词，优化器使用的是AdamOptimizer。
loss = tf.contrib.seq2seq.sequence_loss(logits, targets=targets, weights=weights)
pred = tf.argmax(logits, axis=2)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

saver = tf.train.Saver()
train_weights = np.ones(shape=[batch_size, sequence_length], dtype=np.float32)

# 创建会话开始执行，这里会用到tf.train.Saver对象来保存和读取模型，保险起见可以每隔一定
# 间隔保存一次模型，下次重启会接着训练而不用从头重新来过，这里因为是一个例子，QA对数量不多，
# 所以直接一次性当成一批送进去训练，而并没有分成多批。
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    epoch = 0
    while epoch < 5000000:
        epoch = epoch + 1
        print("epoch:", epoch)
        for step in range(0, 1):
            print("step:", step)
            train_x, train_y, train_target = loadQA()
            train_encoder_inputs = train_x[step * batch_size:step * batch_size + batch_size, :]
            train_decoder_inputs = train_y[step * batch_size:step * batch_size + batch_size, :]
            train_targets = train_target[step * batch_size:step * batch_size + batch_size, :]
            op = sess.run(train_op, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
                                               weights: train_weights, decoder_inputs: train_decoder_inputs})
            cost = sess.run(loss, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
                                             weights: train_weights, decoder_inputs: train_decoder_inputs})
            print(cost)
            step = step + 1
        if epoch % 100 == 0:
            saver.save(sess, model_dir + '/model.ckpt', global_step=epoch + 1)

