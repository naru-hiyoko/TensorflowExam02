import tensorflow as tf
from ops import fc, Convolution2d, relu, batch_norm, pool2d

class VGG(object):


    def __init__(self):
        pass

    def compile(self, x, reuse=False, withDict=False):

        net = {}

        conv1_1 = Convolution2d(1, 32, reuse=reuse)
        conv1_2 = Convolution2d(32, 32, stride=[2, 2, 2, 2], reuse=reuse)

        conv2_1 = Convolution2d(32, 64, reuse=reuse)
        conv2_2 = Convolution2d(64, 64, reuse=reuse)

        h1 = batch_norm(relu(conv1_1.conv2d(x, "c1")), "bn1")
        h2 = batch_norm(relu(conv1_2.conv2d(h1, "c2")), "bn2")
        h3 = batch_norm(relu(conv2_1.conv2d(h2, "c3")), "bn3")
        h4 = batch_norm(relu(conv2_2.conv2d(h3, "c4")), "bn4")


        shape = h4.get_shape()
        count = shape[1] * shape[2] * shape[3]
        _h4 = tf.reshape(h4, [-1, int(count)])

        h5 = batch_norm(relu(fc(_h4, count, 4200, "f1", reuse=reuse)), "bn5")
        h6 = fc(h5, 4200, 10, "fc2", reuse=reuse)

        net["h1"] = h1
        net["h2"] = h2
        net["h3"] = h3
        net["h4"] = h4
        net["h6"] = h6

        if not reuse:
            if withDict:
                return net
            else:
                return h6
        else:
            return net
