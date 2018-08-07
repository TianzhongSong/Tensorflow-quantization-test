from utils.layers import *
import tensorflow as tf
import numpy as np


def quantize(weights):
    abs_weights = np.abs(weights)
    vmax = np.max(abs_weights)
    s = vmax / 127.
    qweights = weights / s
    qweights = np.round(qweights)
    qweights = qweights.astype(np.int8)
    return qweights, s


def get_weights_biases(weights, weight_name, bias_name='bbb', quant=True):
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


def VGG16(x, weights):
    x = tf.reshape(x, shape=[-1, 224, 224, 3])

    # block 1
    w, b, s = get_weights_biases(weights, 'block1_conv1_W_1:0', 'block1_conv1_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')

    w, b, s = get_weights_biases(weights, 'block1_conv2_W_1:0', 'block1_conv2_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')
    x = maxpool_2d(x, k=2, s=2)

    # block2
    w, b, s = get_weights_biases(weights, 'block2_conv1_W_1:0', 'block2_conv1_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')

    w, b, s = get_weights_biases(weights, 'block2_conv2_W_1:0', 'block2_conv2_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')
    x = maxpool_2d(x, k=2, s=2)

    # block3
    w, b, s = get_weights_biases(weights, 'block3_conv1_W_1:0', 'block3_conv1_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')

    w, b, s = get_weights_biases(weights, 'block3_conv2_W_1:0', 'block3_conv2_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')

    w, b, s = get_weights_biases(weights, 'block3_conv3_W_1:0', 'block3_conv3_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')
    x = maxpool_2d(x, k=2, s=2)

    # block4
    w, b, s = get_weights_biases(weights, 'block4_conv1_W_1:0', 'block4_conv1_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')

    w, b, s = get_weights_biases(weights, 'block4_conv2_W_1:0', 'block4_conv2_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')

    w, b, s = get_weights_biases(weights, 'block4_conv3_W_1:0', 'block4_conv3_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')
    x = maxpool_2d(x, k=2, s=2)

    # block5
    w, b, s = get_weights_biases(weights, 'block5_conv1_W_1:0', 'block5_conv1_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')

    w, b, s = get_weights_biases(weights, 'block5_conv2_W_1:0', 'block5_conv2_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')

    w, b, s = get_weights_biases(weights, 'block5_conv3_W_1:0', 'block5_conv3_b_1:0')
    x = conv_2d(x, w, b, s, activation='relu')
    x = maxpool_2d(x, k=2, s=2)

    # fc1
    w, b, s = get_weights_biases(weights, 'fc1_W_1:0', 'fc1_b_1:0')
    x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
    x = denselayer(x, w, b, s, activation='relu')

    # fc2
    w, b, s = get_weights_biases(weights, 'fc2_W_1:0', 'fc2_b_1:0')
    x = denselayer(x, w, b, s, activation='relu')

    # fc3
    w, b, s = get_weights_biases(weights, 'predictions_W_1:0', 'predictions_b_1:0')
    x = denselayer(x, w, b, s)
    return x
