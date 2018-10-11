import tensorflow as tf

import numpy as np
from tensorflow_study.word_id_test import Word_Id_Map

# 预测阶段也同样要创建相同的模型，然后将训练时保存的模型加载进来，然后实现对问题的回答的预测。
# 预测阶段我们用cpu来执行就行了，避免使用GPU。创建图的步骤和训练时基本一致，参数也要保持一致，
# 不同的地方在于我们要将embedding_rnn_seq2seq函数的feed_previous参数设为True,
# 因为我们已经没有decoder输入了。另外我们也不需要损失函数和优化器，仅仅提供预测函数即可。

# 创建会话后开始执行，先加载model目录下的模型，然后再将待测试的问题转成向量形式，
# 接着进行预测，得到输出如下：
# [‘how’, ‘do’, ‘you’, ‘do’, ‘this’, ‘’, ‘’, ‘’, ‘’, ‘’]

with tf.device('/cpu:0'):
    batch_size = 1
    sequence_length = 20
    num_encoder_symbols = 1004
    num_decoder_symbols = 1004
    embedding_size = 256
    hidden_size = 256
    num_layers = 2

    encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
    decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])

    targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
    weights = tf.placeholder(dtype=tf.float32, shape=[batch_size, sequence_length])

    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    results, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        tf.unstack(encoder_inputs, axis=1),
        tf.unstack(decoder_inputs, axis=1),
        cell,
        num_encoder_symbols,
        num_decoder_symbols,
        embedding_size,
        feed_previous=True,
    )
    logits = tf.stack(results, axis=1)
    pred = tf.argmax(logits, axis=2)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, module_file)
        map = Word_Id_Map()
        encoder_input = map.sentence2ids(['how', 'do', 'you', 'do', 'this'])
        # encoder_input = map.sentence2ids(
        #     ['i', 'think', 'he', 'is', 'an', 'old', 'pupper', 'ian', 'he', 'just', 'wants', 'some', 'pizza', 'tbh'])

        encoder_input = encoder_input + [3 for i in range(0, 20 - len(encoder_input))]
        encoder_input = np.asarray([np.asarray(encoder_input)])
        decoder_input = np.zeros([1, 20])
        print('encoder_input : ', encoder_input)
        print('decoder_input : ', decoder_input)
        pred_value = sess.run(pred, feed_dict={encoder_inputs: encoder_input, decoder_inputs: decoder_input})
        print(pred_value)
        sentence = map.ids2sentence(pred_value[0])
        print(sentence)


