import nltk
import itertools
import numpy as np
import pickle

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

FILENAME = './data/conv.txt'

# 训练样本集
# 主要是一些QA对，开放数据也很多可以下载，这里只是随便选用一小部分问题和回答，
# 存放的格式是第一行为问题，第二行为回答，第三行又是问题，第四行为回答，以此类推。
#
# 数据预处理
# 要训练就肯定要将数据转成数字，可以用0到n的值来表示整个词汇，每个值表示一个单词，
# 这里用VOCAB_SIZE来定义。还有问题的最大最小长度，回答的最大最小长度。除此之外还要定义
# UNK、GO、EOS和PAD符号，分别表示未知单词，比如你超过 VOCAB_SIZE范围的则认为未知单词，
# GO表示decoder开始的符号，EOS表示回答结束的符号，而PAD用于填充，因为所有QA对放到同个
# seq2seq模型中输入和输出都必须是相同的，于是就需要将较短长度的问题或回答用PAD进行填充。

limit = {
    'maxq': 20,
    'minq': 0,
    'maxa': 18,
    'mina': 3
}

UNK = 'unk'
GO = '<go>'
EOS = '<eos>'
PAD = '<pad>'
VOCAB_SIZE = 1000


def read_lines(filename):
    return open(filename, encoding='UTF-8').read().split('\n')


def split_line(line):
    return line.split('.')


def filter_line(line, whitelist):
    return ''.join([ch for ch in line if ch in whitelist])


# 我们还要得到整个语料库所有单词的频率统计，还要根据频率大小统计出排名前n个频率的单词
# 作为整个词汇，也就是前面对应的VOCAB_SIZE。另外我们还需要根据索引值得到单词的索引，
# 还有根据单词得到对应索引值的索引。
def index_(tokenized_sentences, vocab_size):
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    index2word = [GO] + [EOS] + [UNK] + [PAD] + [x[0] for x in vocab]
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    return index2word, word2index, freq_dist


# 按照QA长度的限制进行筛选。
def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences) // 2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i + 1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i + 1])
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len) * 100 / raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


# 前面也说到在我们的seq2seq模型中，对于encoder来说，问题的长短是不同的，
# 那么不够长的要用PAD进行填充，比如问题为”how are you”，假如长度定为10，
# 则需要将其填充为”how are you pad pad pad pad pad pad pad”。对于decoder来说，
# 要以GO开始，以EOS结尾，不够长还得填充，比如”fine thank you”，则要处理成”
# go fine thank you eos pad pad pad pad pad “。第三个要处理的则是我们的target，
# target其实和decoder的输入是相同的，只不过它刚好有一个位置的偏移，比如上面要去掉go，
# 变成”fine thank you eos pad pad pad pad pad pad”。
def zero_pad(qtokenized, atokenized, w2idx):
    # 总共有多少个问题句子
    data_len = len(qtokenized)
    # +2 dues to '<go>' and '<eos>'
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa'] + 2], dtype=np.int32)
    idx_o = np.zeros([data_len, limit['maxa'] + 2], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'], 1)
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], 2)
        o_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], 3)
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)
        idx_o[i] = np.array(o_indices)

    return idx_q, idx_a, idx_o


def pad_seq(seq, lookup, maxlen, flag):
    if flag == 1:
        indices = []
    elif flag == 2:
        indices = [lookup[GO]]
    elif flag == 3:
        indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    if flag == 1:
        return indices + [lookup[PAD]] * (maxlen - len(seq))
    elif flag == 2:
        return indices + [lookup[EOS]] + [lookup[PAD]] * (maxlen - len(seq))
    elif flag == 3:
        return indices + [lookup[EOS]] + [lookup[PAD]] * (maxlen - len(seq) + 1)


def process_data():
    print('\n>> Read lines from file')
    lines = read_lines(filename=FILENAME)
    lines = [line.lower() for line in lines]
    print('\n>> Filter lines')
    lines = [filter_line(line, EN_WHITELIST) for line in lines]
    print('\n>> 2nd layer of filtering')
    qlines, alines = filter_data(lines)
    print('\nq : {0} '.format(qlines[:]))
    print('\na : {0}'.format(alines[:]))
    print('\n>> Segment lines into words')
    qtokenized = [wordlist.split(' ') for wordlist in qlines]
    atokenized = [wordlist.split(' ') for wordlist in alines]
    print('\n:: Sample from segmented list of words')
    print('\nq : {0}'.format(qtokenized[:]))
    print('\na : {0}'.format(atokenized[:]))

    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_(qtokenized + atokenized, vocab_size=VOCAB_SIZE)
    print(idx2w)
    with open('idx2w.pkl', 'wb') as f:
        pickle.dump(idx2w, f)
    with open('w2idx.pkl', 'wb') as f:
        pickle.dump(w2idx, f)
    idx_q, idx_a, idx_o = zero_pad(qtokenized, atokenized, w2idx)

    print('\nq : {0} ; a : {1} : o : {2}'.format(idx_q[0], idx_a[0], idx_o[0]))
    print('\nq : {0} ; a : {1} : o : {2}'.format(idx_q[1], idx_a[1], idx_o[1]))

    print('\n >> Save numpy arrays to disk')
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)
    np.save('idx_o.npy', idx_o)

    metadata = {
        'w2idx': w2idx,
        'idx2w': idx2w,
        'limit': limit,
        'freq_dist': freq_dist
    }

    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def load_data(PATH='.'):
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a


if __name__ == '__main__':
    # 然后将上面处理后的结构都持久化起来，供训练时使用。
    process_data()


