from net import VGG
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import sys

import dataset

if __name__ == '__main__':

    print("load dataset...")
    m = dataset.SingleFontData()


    net  = VGG()
    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
    y = net.compile(x)
    t = tf.placeholder(tf.float32, [None, 10])
    _y = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y)


    entropy = tf.reduce_mean(_y, reduction_indices=[0])
    optimizer = tf.train.AdagradOptimizer(0.01).minimize(entropy)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    print("restore model...")
    tf.train.Saver().restore(session, "./discriminator/model")

    """
    model = net.compile(x, reuse=True)


    initial = tf.random_normal([1, 64, 64, 1]) * 0.256
    _in = tf.Variable(initial)

    model["h2"].eval(feed_dict={
        x: _in
    })

    print("ok")
    """

    print("training loop...")
    for i in range(21):
        xs, ys = m.train.next_batch(10)

        _in = np.reshape(xs, [-1, 64, 64, 1])
        session.run(optimizer,
                    feed_dict= {
                        x: _in, t: ys
                    })
        if i % 5 == 0 :
            _loss = session.run(entropy, feed_dict={
                x: _in, t: ys
            })
            print(_loss)

    ### test sequence
    acc_graph = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(t, 1)), tf.float32))

    _acc = session.run(acc_graph, feed_dict={
        x: np.reshape(m.test.images, [-1, 64, 64, 1]), t: m.test.labels
    })
    print('acc: {}'.format(_acc))
    tf.train.Saver().save(session, "./discriminator/model")


    print("done!")
