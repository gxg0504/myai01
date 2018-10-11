import numpy as np
from sklearn.datasets import load_sample_images
import tensorflow as tf
import matplotlib.pyplot as plt


# 加载数据集
# 输入图片通常是3D，[height, width, channels]
# mini-batch通常是4D，[mini-batch size, height, width, channels]
dataset = np.array(load_sample_images().images, dtype=np.float32)
# 数据集里面两张图片，一个中国庙宇，一个花
batch_size, height, width, channels = dataset.shape
print(batch_size, height, width, channels)

# 创建输入和一个池化层
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
# TensorFlow不支持池化多个实例，所以ksize的第一个batch size是1
# TensorFlow不支持池化同时发生的长宽高，所以必须有一个是1，这里channels就是depth维度为1
max_pool = tf.nn.avg_pool(X, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
# avg_pool()

with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})

plt.imshow(dataset[0].astype(np.uint8))
plt.show()
plt.imshow(output[0].astype(np.uint8))  # 画输入的第一个图像
plt.show()
