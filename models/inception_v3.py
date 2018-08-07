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


def get_weights(weights, weight_name, bias_name='bbb', quant=True):
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


def get_bn_param(weights, mean, std, beta):
    mean = tf.constant(weights[mean], dtype=tf.float32)
    std = tf.constant(weights[std], dtype=tf.float32)
    beta = tf.constant(weights[beta], dtype=tf.float32)
    return mean, std, beta


def conv2d_bn(x, layer_count, weights, strides=1, padding='SAME'):
    bn_beta = 'batch_normalization_' + str(layer_count) + '/beta:0'
    bn_mean = 'batch_normalization_' + str(layer_count) + '/moving_mean:0'
    bn_var = 'batch_normalization_' + str(layer_count) + '/moving_variance:0'
    conv_name = 'conv2d_' + str(layer_count) + '/kernel:0'
    bias_name = 'conv2d_' + str(layer_count) + '/bias:0'

    layer_count += 1
    w, b, s = get_weights(weights, conv_name, bias_name)
    x = conv_2d(x, w, b, s, strides=strides, padding=padding)

    mean, std, beta = get_bn_param(weights, bn_mean, bn_var, bn_beta)
    x = batch_norm(x, mean, std, beta)
    x = tf.nn.relu(x)
    return x, layer_count


def InceptionV3(img_input, weights):
    layer_count = 1

    x = tf.reshape(img_input, shape=[-1, 299, 299, 3])

    x, layer_count = conv2d_bn(x, layer_count, weights, strides=2, padding='VALID')
    x, layer_count = conv2d_bn(x, layer_count, weights, strides=1, padding='VALID')
    x, layer_count = conv2d_bn(x, layer_count, weights)
    x = maxpool_2d(x, k=3, s=2, padding='SAME')

    x, layer_count = conv2d_bn(x, layer_count, weights, strides=1, padding='VALID')
    x, layer_count = conv2d_bn(x, layer_count, weights, strides=1, padding='VALID')
    x = maxpool_2d(x, k=3, s=2, padding='SAME')

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1, layer_count = conv2d_bn(x, layer_count, weights)

    branch5x5, layer_count = conv2d_bn(x, layer_count, weights)
    branch5x5, layer_count = conv2d_bn(branch5x5, layer_count, weights)

    branch3x3dbl, layer_count = conv2d_bn(x, layer_count, weights)
    branch3x3dbl, layer_count = conv2d_bn(branch3x3dbl, layer_count, weights)
    branch3x3dbl, layer_count = conv2d_bn(branch3x3dbl, layer_count, weights)

    branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
    branch_pool, layer_count = conv2d_bn(branch_pool, layer_count, weights)

    x = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3)

    # mixed 1: 35 x 35 x 256
    branch1x1, layer_count = conv2d_bn(x, layer_count, weights)

    branch5x5, layer_count = conv2d_bn(x, layer_count, weights)
    branch5x5, layer_count = conv2d_bn(branch5x5, layer_count, weights)

    branch3x3dbl, layer_count = conv2d_bn(x, layer_count, weights)
    branch3x3dbl, layer_count = conv2d_bn(branch3x3dbl, layer_count, weights)
    branch3x3dbl, layer_count = conv2d_bn(branch3x3dbl, layer_count, weights)

    branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
    branch_pool, layer_count = conv2d_bn(branch_pool, layer_count, weights)

    x = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3)

    # mixed 2: 35 x 35 x 256
    branch1x1, layer_count = conv2d_bn(x, layer_count, weights)

    branch5x5, layer_count = conv2d_bn(x, layer_count, weights)
    branch5x5, layer_count = conv2d_bn(branch5x5, layer_count, weights)

    branch3x3dbl, layer_count = conv2d_bn(x, layer_count, weights)
    branch3x3dbl, layer_count = conv2d_bn(branch3x3dbl, layer_count, weights)
    branch3x3dbl, layer_count = conv2d_bn(branch3x3dbl, layer_count, weights)

    branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
    branch_pool, layer_count = conv2d_bn(branch_pool, layer_count, weights)
    x = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3)

    # mixed 3: 17 x 17 x 768
    branch3x3, layer_count = conv2d_bn(x, layer_count, weights, strides=2, padding='VALID')

    branch3x3dbl, layer_count = conv2d_bn(x, layer_count, weights)
    branch3x3dbl, layer_count = conv2d_bn(branch3x3dbl, layer_count, weights)
    branch3x3dbl, layer_count = conv2d_bn(branch3x3dbl, layer_count, weights, strides=2, padding='VALID')

    branch_pool = maxpool_2d(x, k=3, s=2, padding='VALID')
    x = tf.concat([branch3x3, branch3x3dbl, branch_pool], axis=3)

    # mixed 4: 17 x 17 x 768
    branch1x1, layer_count = conv2d_bn(x, layer_count, weights)

    branch7x7, layer_count = conv2d_bn(x, layer_count, weights)
    branch7x7, layer_count = conv2d_bn(branch7x7, layer_count, weights)
    branch7x7, layer_count = conv2d_bn(branch7x7, layer_count, weights)

    branch7x7dbl, layer_count = conv2d_bn(x, layer_count, weights)
    branch7x7dbl, layer_count = conv2d_bn(branch7x7dbl, layer_count, weights)
    branch7x7dbl, layer_count = conv2d_bn(branch7x7dbl, layer_count, weights)
    branch7x7dbl, layer_count = conv2d_bn(branch7x7dbl, layer_count, weights)
    branch7x7dbl, layer_count = conv2d_bn(branch7x7dbl, layer_count, weights)

    branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
    branch_pool, layer_count = conv2d_bn(branch_pool, layer_count, weights)
    x = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3)

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1, layer_count = conv2d_bn(x, layer_count, weights)

        branch7x7, layer_count = conv2d_bn(x, layer_count, weights)
        branch7x7, layer_count = conv2d_bn(branch7x7, layer_count, weights)
        branch7x7, layer_count = conv2d_bn(branch7x7, layer_count, weights)

        branch7x7dbl, layer_count = conv2d_bn(x, layer_count, weights)
        branch7x7dbl, layer_count = conv2d_bn(branch7x7dbl, layer_count, weights)
        branch7x7dbl, layer_count = conv2d_bn(branch7x7dbl, layer_count, weights)
        branch7x7dbl, layer_count = conv2d_bn(branch7x7dbl, layer_count, weights)
        branch7x7dbl, layer_count = conv2d_bn(branch7x7dbl, layer_count, weights)

        branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
        branch_pool, layer_count = conv2d_bn(branch_pool, layer_count, weights)
        x = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3)

    # mixed 7: 17 x 17 x 768
    branch1x1, layer_count = conv2d_bn(x, layer_count, weights)

    branch7x7, layer_count = conv2d_bn(x, layer_count, weights)
    branch7x7, layer_count = conv2d_bn(branch7x7, layer_count, weights)
    branch7x7, layer_count = conv2d_bn(branch7x7, layer_count, weights)

    branch7x7dbl, layer_count = conv2d_bn(x, layer_count, weights)
    branch7x7dbl, layer_count = conv2d_bn(branch7x7dbl, layer_count, weights)
    branch7x7dbl, layer_count = conv2d_bn(branch7x7dbl, layer_count, weights)
    branch7x7dbl, layer_count = conv2d_bn(branch7x7dbl, layer_count, weights)
    branch7x7dbl, layer_count = conv2d_bn(branch7x7dbl, layer_count, weights)

    branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
    branch_pool, layer_count = conv2d_bn(branch_pool, layer_count, weights)
    x = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3)

    # mixed 8: 8 x 8 x 1280
    branch3x3, layer_count = conv2d_bn(x, layer_count, weights)
    branch3x3, layer_count = conv2d_bn(branch3x3, layer_count, weights, strides=2, padding='VALID')

    branch7x7x3, layer_count = conv2d_bn(x, layer_count, weights)
    branch7x7x3, layer_count = conv2d_bn(branch7x7x3, layer_count, weights)
    branch7x7x3, layer_count = conv2d_bn(branch7x7x3, layer_count, weights)
    branch7x7x3, layer_count = conv2d_bn(branch7x7x3, layer_count, weights, strides=2, padding='VALID')

    branch_pool = maxpool_2d(x, k=3, s=2, padding='VALID')
    x = tf.concat([branch3x3, branch7x7x3, branch_pool], axis=3)

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1, layer_count = conv2d_bn(x, layer_count, weights)

        branch3x3, layer_count = conv2d_bn(x, layer_count, weights)
        branch3x3_1, layer_count = conv2d_bn(branch3x3, layer_count, weights)
        branch3x3_2, layer_count = conv2d_bn(branch3x3, layer_count, weights)
        branch3x3 = tf.concat([branch3x3_1, branch3x3_2], axis=3)

        branch3x3dbl, layer_count = conv2d_bn(x, layer_count, weights)
        branch3x3dbl, layer_count = conv2d_bn(branch3x3dbl, layer_count, weights)
        branch3x3dbl_1, layer_count = conv2d_bn(branch3x3dbl, layer_count, weights)
        branch3x3dbl_2, layer_count = conv2d_bn(branch3x3dbl, layer_count, weights)
        branch3x3dbl = tf.concat([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
        branch_pool, layer_count = conv2d_bn(branch_pool, layer_count, weights)
        x = tf.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3)

    x = avgpool_2d(x, k=8)
    w, b, s = get_weights(weights, 'predictions/kernel:0', 'predictions/bias:0')
    x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
    x = denselayer(x, w, b, s)

    return x
