# 1,初始化H0？
# 2,除了计算Y3，还可以怎么使用最后时刻的H3？

import zipfile
import random
from mxnet import nd

with zipfile.ZipFile("./jaychou_lyrics.txt.zip", "r") as zin:
    zin.extractall("./jaychou_lyrics/")

with open("./jaychou_lyrics/jaychou_lyrics.txt", encoding='utf-8') as f:
    corpus_chars = f.read()
print(corpus_chars[0:49])
# 64925多汉字
print(len(corpus_chars))
# 接着稍微处理下数据集，为了打印方便，换行符替换成空格，然后截去后面一段使得训练快一些
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:20000]

# 字符的数值表示
# 先把数据里面所有不同的字符拿出来做成一个字典
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

vocab_size = len(char_to_idx)
print('vocab size:', vocab_size)

# 然后可以把每个字符转成从0开始的索引（index）来方便之后的使用
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:40]
print('char: \n', ''.join([idx_to_char[idx] for idx in sample]))
print('\nindices: \n', sample)

# 时序数据的批量采样
# 同之前一样我们需要每次随机读取batch_size个样本和其对应的标号
# 这里的样本和前面有点不一样，这里一个样本通常包含一系列连续的字符，前馈神经网络里每个字符作为一个样本
# 如果我们把序列长度num_steps设成5，那么一个可能的样本是"想要有直升"
# 其对应的标号仍然是长度为5的序列，每个字符是对应样本里面字符后面的那个
# 例如前面样本的标号就是"要有直升机"


# 随机批量采样
# 下面代码每次从数据里随机采样一个批量
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    """
    随机采样
    :param corpus_indices: 原始文本
    :param batch_size: 每批次里放多少个序列数据
    :param num_steps: 每个序列数据的长度
    :param ctx:
    :return:
    """
    # 减一是为了label的索引是相应的data的索引加一
    # 把整体文本不重复的切成多少份
    num_examples = (len(corpus_indices) - 1) // num_steps
    # 运行多少轮次才能把这么多的样本学习完
    epoch_size = num_examples // batch_size
    # 随机化样本
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回num_steps个序列数据,即一条样本生成
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        # 拿到每批数据是第几个样本
        batch_indices = example_indices[i: i + batch_size]
        # 找到每批数据中每个样本具体的开始位置，然后截取数据
        data = nd.array(
            [_data(j * num_steps) for j in batch_indices], ctx=ctx
        )
        label = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx
        )
        yield data, label


# 为了便于理解时序数据上的随机批量采样，让我们输入一个从0到29的人工序列，看下读出来长什么样
# 相当于假设序列总长度为30，序列中就直接放数字0到29，测试上面随机采样函数
my_seq = list(range(30))
for data, label in data_iter_random(my_seq, batch_size=2, num_steps=3):
    print('data: ', data, "\nlabel:", label, '\n')

print('='*30)

# 随机采样的问题是，批次和批次之间样本不相关，每次需要初始化隐藏状态state
# 说白了就是丢了一些相邻关系，1和2之间，2和3之间的关系等等
# 第二种采样方法，相邻批量采样，好处是上一次的状态h3可以作为下次批量时候的h0
# 相邻批量采样，除了对原序列做随机批量采样之外，我们还可以使用相邻的两个随机批量在原始序列上的位置相毗邻
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    """
    相邻批量采样
    :param corpus_indices: 原始文本
    :param batch_size: 每个批次数据条数
    :param num_steps: 每个批次每条数据序列长度
    :param ctx:
    """
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    # 计算整体分割batch_size个块后的数据长度
    batch_len = data_len // batch_size
    # reshape转换为行为每个批次多少条，列为每个批次长度*多少个批次
    indices = corpus_indices[0: batch_size * batch_len].reshape((
        batch_size, batch_len
    ))
    # 计算需要的批次数量可以把数据学习完
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        i = i * num_steps
        data = indices[:, i: i + num_steps]
        label = indices[:, i + 1: i + num_steps + 1]
        yield data, label


# 前面我们每次得到的数据即每批次是一个batch_size*num_steps的批量，下面这个函数
# 将其转换成num_steps个可以输入进网络的batch_size*vocab_size的矩阵
def get_inputs(data):
    return [nd.one_hot(X, vocab_size) for X in data.T]


my_seq = list(range(30))
for data, label in data_iter_consecutive(my_seq, batch_size=2, num_steps=3):
    print('data: ', data, '\nlabel:', label, '\n')
    inputs = get_inputs(data)
    print('input length: ', len(inputs))
    print('input[0] shape: ', inputs[0].shape)

# one-hot编码
# 注意到每个字符现在是用一个整数来表示，而输入进网络我们需要一个定长的向量
print(nd.one_hot(nd.array([0, 2]), vocab_size))

# 初始化模型参数
import mxnet as mx

hidden_size = 256
std = 0.01


def get_params():
    # 隐含层
    W_xh = nd.random_normal(scale=std, shape=(vocab_size, hidden_size))
    W_hh = nd.random_normal(scale=std, shape=(hidden_size, hidden_size))
    b_h = nd.zeros(hidden_size)

    # 输出层
    W_hy = nd.random_normal(scale=std, shape=(hidden_size, vocab_size))
    b_y = nd.zeros(vocab_size)

    params = [W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params


# 定义模型
# 当序列中某一个时间戳的输入为一个样本数为batch_size的批量，而整个序列长度为num_step时，以下rnn函数的
# inputs和outputs均为num_steps个尺寸batch_size*vocab_size的矩阵
# 隐含状态变量H是一个尺寸为batch_size*hidden_size的矩阵
def rnn(inputs, H, W_xh, W_hh, b_h, W_hy, b_y):
    # inputs: num_steps个尺寸为batch_size*vocab_size矩阵
    # H: 尺寸为batch_size * hidden_size矩阵
    # outputs: num_steps个尺寸为batch_size*vocab_size矩阵
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return outputs, H


