import collections
import numpy as np
import tensorflow as tf

# -------------------------------数据预处理---------------------------#

poetry_file = './data/poetry.txt'

# 诗集
poetrys = []
with open(poetry_file, "r", encoding='utf-8', ) as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = '[' + content + ']'
            poetrys.append(content)
        except Exception as e:
            pass

# 按诗的字数排序
poetrys = sorted(poetrys, key=lambda line: len(line))
print('唐诗总数: ', len(poetrys))

# 统计每个字出现次数
all_words = []
count = 0
for poetry in poetrys:
    temp = [word for word in poetry]
    all_words += temp
    # if count < 5:
    #     print(temp)
    #     print(all_words)
    count += 1
counter = collections.Counter(all_words)
print(counter.items())
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
print(count_pairs)
print(*count_pairs)
words, _ = zip(*count_pairs)
# 倒序排序后取出了所有字
print(words)

# 取前多少个常用字,并且加上空格
print(len(words))
words = words[:len(words)] + (' ',)
print(words)
print(len(words))

# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))
print(word_num_map)

# 把诗转换为向量形式
# 定义一个查索引的方法，如果是常用字就给index，如果不是就给默认值len(words)
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]
# [[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
# [339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
# ....]

# 每次取256首诗进行训练
batch_size = 256
# 计算多少次可以把诗学完了
n_chunk = len(poetrys_vector) // batch_size
# 准备数据
x_batches = []
y_batches = []
for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size
    # 每次取256首诗
    batches = poetrys_vector[start_index:end_index]
    # 计算256首诗里面最长的长度
    length = max(map(len, batches))
    # 创建全部为空格的索引号的矩阵
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    # 把每首诗的向量覆盖填入
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:, :-1] = xdata[:, 1:]
    
    # xdata             ydata
    # [6,2,4,6,9]       [2,4,6,9,9]
    # [1,4,2,8,5]       [4,2,8,5,5]
    
    x_batches.append(xdata)
    y_batches.append(ydata)

# ---------------------------------------RNN--------------------------------------#

input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])


# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
    if model == 'rnn':
        cell_fun = tf.nn.rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    # 单个节点里面神经网络有两层，堆叠的，相当于网络层更深
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope('rnnlm'):
        # len(words) + 1 加一的原因是句子向量里面有一个不在所有词里面的默认值
        # 构建从Cell单元输出得结果到输出层Y得W和b矩阵
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words) + 1])
        softmax_b = tf.get_variable("softmax_b", [len(words) + 1])
        # 是来构建X输入到Cell之间得变化，说白了就是把X变成X_in交给RNN Cell单元
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [len(words) + 1, rnn_size])
            # 相当于对每个词进行one-hot编码或者词向量生成
            inputs = tf.nn.embedding_lookup(embedding, input_data)

    # 下面一行是来构建RNN网络拓扑结构
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs, [-1, rnn_size])
    # 计算从Cell单元输出的结果到输出层Y的结果
    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    return logits, last_state, probs, cell, initial_state


# 训练
def train_neural_network():
    logits, last_state, _, _, _ = neural_network()
    targets = tf.reshape(output_targets, [-1])
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)],
                                                  len(words))
    cost = tf.reduce_mean(loss)
    learning_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    # Gradient Clipping的引入是为了处理gradient explosion或者gradients vanishing的问题。当在一次迭代中权重的更新过于迅猛的话，
    # 很容易导致loss divergence。Gradient Clipping的直观作用就是让权重的更新限制在一个合适的范围。
    # clip_norm是截取的比率, 这个函数返回截取过的梯度张量
    # minimize() = compute_gradients() + apply_gradients()
    # 这里相当于将计算梯度和更新梯度变成两部分来做
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())

        for epoch in range(50):
            sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
            n = 0
            for batch in range(n_chunk):
                train_loss, _, _ = sess.run([cost, last_state, train_op],
                                            feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})
                n += 1
                print(epoch, batch, train_loss)
            if epoch % 7 == 0:
                saver.save(sess, './poetry.module', global_step=epoch)


train_neural_network()


