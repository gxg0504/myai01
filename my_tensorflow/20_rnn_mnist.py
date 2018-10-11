import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# 创建计算上下文环境
sess = tf.Session()
# 读取mnist数据集，one_hot=True将y列编码为维度10分类的0，1编码
mnist = input_data.read_data_sets('my_mnist_rnn', one_hot=True)
# 打印输出训练集的形状(55000, 784)
print(mnist.train.images.shape)

# 超参数
lr = 0.001
training_iters = 100000
batch_size = 128
n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

# 图输入
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 定义权重
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # 隐藏层输入到单元
    # X(128 batch, 28 steps, 28 inputs)
    # reshape之后(128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in ==> (128 batch* 28 steps, 128 hidden)
    # 这行的X*W+b的运算是topo结构里面的整体的X和整体Win、bin的计算
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 单元
    # forget_bias = 1.0 是初始值
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell 分为两部分 (c_state, m_state), RNN会计算每一个cell里面的结果
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # states里面(c_state, m_state)，如果只是一般的RNN的话，就只是m_state
    # states是最后一个state，outputs是个list，每一步的output都存在里面
    # 有rnn和dynamic_rnn，区别是每批次的维度可以不一样，rnn必须一样
    # 28 steps就是我们的time轴，time_major是不是第一个维度，我们的是在维度为2的地方，所有False
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # 隐藏层到输入结果
    results = tf.matmul(states[1], weights['out']) + biases['out']

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        test_batch_xs, test_batch_ys = mnist.test.next_batch(batch_size)
        test_batch_xs = test_batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: test_batch_xs,
                y: test_batch_ys
            }))
        step += 1


