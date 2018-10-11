import numpy as np
import tensorflow as tf
from pprint import pprint
from tensorflow.python.framework import tensor_shape
from tensorflow.contrib.rnn import BasicRNNCell

from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder, BasicDecoderOutput
from tensorflow.contrib.seq2seq.python.ops.helper import TrainingHelper
from tensorflow.python.layers.core import Dense


sequence_length = [3, 4, 3, 1, 0]
batch_size = 5
max_time = 8
input_size = 7
hidden_size = 10
output_size = 3

inputs = np.random.randn(batch_size, max_time, input_size).astype(np.float32)

output_layer = Dense(output_size) # will get a trainable variable size [hidden_size x output_size]

print(output_layer.__dict__) # doesn't have any variable yet

dec_cell = BasicRNNCell(hidden_size)
helper = TrainingHelper(inputs, sequence_length)

decoder = BasicDecoder(
    cell=dec_cell,
    helper=helper,
    initial_state=dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
    output_layer=output_layer)

print(decoder.__dict__)

print([attr for attr in dir(decoder) if '__' not in attr])
print(decoder.output_size)
print(decoder.output_dtype)
print(decoder.batch_size)


first_finished, first_inputs, first_state = decoder.initialize()
print(first_finished, first_inputs, first_state)
# first_finished: [batch_size]
# first_inputs: [batch_size x input_size]
# first_state: [batch_size x hidden_size]

step_outputs, step_state, step_next_inputs, step_finished = decoder.step(
    tf.constant(0), first_inputs, first_state)
print(step_outputs, step_state, step_next_inputs, step_finished)
# step_outputs.rnn_output: [batch_size x output_size]
# step_outputs.sample_id: [batch_size]
# step_state: [batch_size x max_time]
# step_next_inputs: [batch_size x input_size]
# step_finished: [batch_size]

print(output_layer.__dict__)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    results = sess.run({
        "batch_size": decoder.batch_size,
        "first_finished": first_finished,
        "first_inputs": first_inputs,
        "first_state": first_state,
        "step_outputs": step_outputs,
        "step_state": step_state,
        "step_next_inputs": step_next_inputs,
        "step_finished": step_finished})
pprint(results)

