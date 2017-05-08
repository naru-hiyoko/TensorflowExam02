import tensorflow as tf


def pool2d(x, k=[2, 2, 2, 2], strides=[1, 1, 1, 1]):
    return tf.nn.max_pool(x, ksize=k, strides=strides, padding='SAME')

def relu(x):
    return tf.maximum(x, 0.0)


def batch_norm(x, scope):
    return tf.contrib.layers.batch_norm(x, decay=0.9, scope=scope)

def fc(x, n_in, n_out, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable("W", [n_in, n_out], initializer=tf.random_normal_initializer(stddev=0.05), trainable=True)
        b = tf.get_variable("b", [n_out], tf.float32, initializer=tf.random_normal_initializer(stddev=0.05), trainable=True)
        return tf.matmul(x, W) + b


class Convolution2d(object):
    def __init__(self, n_in, n_out, k = 3, stride=[1, 1, 1, 1], reuse=False):
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.reuse = reuse
        self.stride = [1, 1, 1, 1]

    def conv2d(self, x, scope):
        with tf.variable_scope(scope, reuse=self.reuse):
            W = tf.get_variable("W", [self.k, self.k, self.n_in, self.n_out], tf.float32, initializer=tf.random_normal_initializer(stddev=0.05), trainable=True)
            b = tf.get_variable("b", [self.n_out], tf.float32, initializer=tf.random_normal_initializer(stddev=0.05), trainable=True)
            return tf.nn.conv2d(x, W, self.stride, padding='SAME') + b



