from utils.layers import conv_2d, maxpool_2d
import tensorflow as tf
from tensorflow.python.keras.layers import UpSampling2D, Permute
import numpy as np


def quantize(weights):
    abs_weights = np.abs(weights)
    vmax = np.max(abs_weights)
    s = vmax / 127.
    qweights = weights / s
    qweights = np.round(qweights)
    qweights = qweights.astype(np.int8)
    return qweights, s


def get_weights_biases_scale(weights, weight_name, bias_name='bbb', quant=True):
    w = weights[weight_name]
    if quant:
        w, s = quantize(w)
        w = tf.constant(w, dtype=tf.float32)
    else:
        w = tf.constant(weights[weight_name], dtype=tf.float32)
        s = 0.
    try:
        b = tf.constant(weights[bias_name], dtype=tf.float32)
    except:
        b = None
    return w, b, s


def Unet(inputs, weights, n_classes, input_height, input_width):
    x = tf.reshape(inputs, shape=[-1, input_height, input_width, 3])

    w, b, s = get_weights_biases_scale(weights, 'conv2d_1/kernel:0', 'conv2d_1/bias:0')
    conv1 = conv_2d(x, w, b, s, activation='relu')
    w, b, s = get_weights_biases_scale(weights, 'conv2d_2/kernel:0', 'conv2d_2/bias:0')
    conv1 = conv_2d(conv1, w, b, s, activation='relu')
    pool1 = maxpool_2d(conv1)

    w, b, s = get_weights_biases_scale(weights, 'conv2d_3/kernel:0', 'conv2d_3/bias:0')
    conv2 = conv_2d(pool1, w, b, s, activation='relu')
    w, b, s = get_weights_biases_scale(weights, 'conv2d_4/kernel:0', 'conv2d_4/bias:0')
    conv2 = conv_2d(conv2, w, b, s, activation='relu')
    pool2 = maxpool_2d(conv2)

    w, b, s = get_weights_biases_scale(weights, 'conv2d_5/kernel:0', 'conv2d_5/bias:0')
    conv3 = conv_2d(pool2, w, b, s, activation='relu')
    w, b, s = get_weights_biases_scale(weights, 'conv2d_6/kernel:0', 'conv2d_6/bias:0')
    conv3 = conv_2d(conv3, w, b, s, activation='relu')
    pool3 = maxpool_2d(conv3)

    w, b, s = get_weights_biases_scale(weights, 'conv2d_7/kernel:0', 'conv2d_7/bias:0')
    conv4 = conv_2d(pool3, w, b, s, activation='relu')
    w, b, s = get_weights_biases_scale(weights, 'conv2d_8/kernel:0', 'conv2d_8/bias:0')
    conv4 = conv_2d(conv4, w, b, s, activation='relu')
    pool4 = maxpool_2d(conv4)

    w, b, s = get_weights_biases_scale(weights, 'conv2d_9/kernel:0', 'conv2d_9/bias:0')
    conv5 = conv_2d(pool4, w, b, s, activation='relu')
    w, b, s = get_weights_biases_scale(weights, 'conv2d_10/kernel:0', 'conv2d_10/bias:0')
    conv5 = conv_2d(conv5, w, b, s, activation='relu')
    pool5 = maxpool_2d(conv5)

    up6 = UpSampling2D(size=(2, 2))(pool5)
    up6 = tf.concat([up6, conv5], axis=-1)
    w, b, s = get_weights_biases_scale(weights, 'conv2d_11/kernel:0', 'conv2d_11/bias:0')
    conv6 = conv_2d(up6, w, b, s, activation='relu')
    w, b, s = get_weights_biases_scale(weights, 'conv2d_12/kernel:0', 'conv2d_12/bias:0')
    conv6 = conv_2d(conv6, w, b, s, activation='relu')

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = tf.concat([up7, conv4], axis=-1)
    w, b, s = get_weights_biases_scale(weights, 'conv2d_13/kernel:0', 'conv2d_13/bias:0')
    conv7 = conv_2d(up7, w, b, s, activation='relu')
    w, b, s = get_weights_biases_scale(weights, 'conv2d_14/kernel:0', 'conv2d_14/bias:0')
    conv7 = conv_2d(conv7, w, b, s, activation='relu')

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = tf.concat([up8, conv3], axis=-1)
    w, b, s = get_weights_biases_scale(weights, 'conv2d_15/kernel:0', 'conv2d_15/bias:0')
    conv8 = conv_2d(up8, w, b, s, activation='relu')
    w, b, s = get_weights_biases_scale(weights, 'conv2d_16/kernel:0', 'conv2d_16/bias:0')
    conv8 = conv_2d(conv8, w, b, s, activation='relu')

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = tf.concat([up9, conv2], axis=-1)
    w, b, s = get_weights_biases_scale(weights, 'conv2d_17/kernel:0', 'conv2d_17/bias:0')
    conv9 = conv_2d(up9, w, b, s, activation='relu')
    w, b, s = get_weights_biases_scale(weights, 'conv2d_18/kernel:0', 'conv2d_18/bias:0')
    conv9 = conv_2d(conv9, w, b, s, activation='relu')

    up10 = UpSampling2D(size=(2, 2))(conv9)
    up10 = tf.concat([up10, conv1], axis=-1)
    w, b, s = get_weights_biases_scale(weights, 'conv2d_19/kernel:0', 'conv2d_19/bias:0')
    conv10 = conv_2d(up10, w, b, s, activation='relu')
    w, b, s = get_weights_biases_scale(weights, 'conv2d_20/kernel:0', 'conv2d_20/bias:0')
    conv10 = conv_2d(conv10, w, b, s, activation='relu')

    w, b, s = get_weights_biases_scale(weights, 'conv2d_21/kernel:0', 'conv2d_21/bias:0')
    conv11 = conv_2d(conv10, w, b, s, activation='relu')
    conv11 = tf.reshape(conv11, shape=[n_classes, input_height * input_width])
    conv11 = Permute((2, 1))(conv11)
    return conv11
