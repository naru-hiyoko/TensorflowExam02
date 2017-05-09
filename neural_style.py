import tensorflow as tf
import numpy as np

from net import VGG

from skimage.io import imread, imsave
from skimage.transform import resize
from operator import mul

try:
    reduce
except NameError:
    from functools import reduce


import sys

content_layers = ["h1", "h2"]
style_layers = ["h2", "h3"]
style_images = ["7_ytrytr_28.jpg", "7_ytrytr_29.jpg", "7_ytrytr_36.jpg", "7_ytrytr_37.jpg"]


def sum_content_losses(sess, net, content_img):
    content_loss = tf.zeros()


def load_image(imageName):
    img = imread(imageName, as_grey=True)
    img = resize(img, [64, 64])
    img = np.reshape(img, [1, 64, 64, 1])
    return img

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def stylize(content_img, init_img = None):


    net  = VGG()
    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
    y = net.compile(x)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    print("restore model...")
    tf.train.Saver().restore(session, "./discriminator/model")


    model = net.compile(x, reuse=True)

    # compute content features in feedforward mode
    content_features = dict()
    for lname in content_layers:
        _content = load_image(content_img)
        content_features[lname] = model[lname].eval(session=session,
        feed_dict={
        x: _content
        })

    # compute style features in feedforward mode
    style_features_images = dict()

    for style_img in style_images:
        _style = load_image('./data/img/'+style_img)
        style_features = dict()
        for lname in style_layers:
            features = model[lname].eval(session=session,
            feed_dict = {
            x: _style
            })
            features = np.reshape(features, [-1, features.shape[3]])
            gram = np.matmul(features.T, features) / features.size
            style_features[lname] = gram
        style_features_images[style_img] = style_features

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        _init = tf.random_uniform([1, 64, 64, 1]) * 0.28
        image = tf.Variable(_init)

        model = net.compile(image, reuse=False, withDict=True)

        content_loss = 0.0
        for lname in content_layers:
            content_loss += 0.1 * tf.nn.l2_loss(model[lname] - content_features[lname])
            content_loss /= np.float32(len(content_layers))



        style_loss = 0.0
        """
        for style_img in style_images:
            style_features = style_features_images[style_img]
            for lname in style_layers:
                layer = model[lname]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / size
                style_loss += 3.0 * tf.reduce_mean(tf.abs(gram - style_features[lname])) / style_features[lname].size
                style_loss /= np.float32(len(style_layers))
        """


        for lname in style_layers:
            layer = model[lname]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            
            sf_feats = None
            for style_img in style_images:
                if sf_feats == None:
                    sf_feats = style_features_images[style_img][lname]
                else:
                    sf_feats += style_features_images[style_img][lname] 
            sf_feats /= np.float32(len(style_images))
            style_loss = 3.0 * tf.reduce_mean(tf.abs(gram - sf_feats))
        style_loss /= np.float32(len(style_layers))

        
        """ category loss """
        enc_loss = 0.0
        ids = np.zeros([1, 10])
        ids[0][7] = 1.0
        enc_loss = 3.0 * tf.nn.l2_loss(model["h6"] - ids)
        


        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])


        # overall loss
        loss = content_loss + style_loss + enc_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            i = -1
            while (True):
                i += 1
                sess.run(train_step)
                best = image.eval()
                if loss.eval() < 500:
                    break
                if i % 50 == 0:
                    print(loss.eval())
                    best = image.eval()
                    _out = best.reshape([64, 64]) * 255.0
                    _out = np.clip(_out, 0.0, 1.0)
                    imsave('hoge.jpg', _out)

    pass

if __name__ == '__main__':
    stylize('./data/img/3_ipaexm_12.jpg', None)
