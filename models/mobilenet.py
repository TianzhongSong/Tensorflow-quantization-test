from utils.layers import depthwise_conv2d, batch_norm, conv_2d, avgpool_2d
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


def get_weights(weights, weight_name, bias_name, quant=True):
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


def get_bn_param(weights, mean, std, beta, gamma):
    mean = tf.constant(weights[mean], dtype=tf.float32)
    std = tf.constant(weights[std], dtype=tf.float32)
    beta = tf.constant(weights[beta], dtype=tf.float32)
    gamma = tf.constant(weights[gamma], dtype=tf.float32)
    return mean, std, beta, gamma


def _depthwise_conv_block(x, weights, strides=1, block_id=0):
    '''
    depthwise convolution and pointwise convolution
    '''
    # depthwise convolution
    bn_beta = 'conv_dw_' + str(block_id) + '_bn/beta:0'
    bn_gamma = 'conv_dw_' + str(block_id) + '_bn/gamma:0'
    bn_mean = 'conv_dw_' + str(block_id) + '_bn/moving_mean:0'
    bn_var = 'conv_dw_' + str(block_id) + '_bn/moving_variance:0'
    conv_name = 'conv_dw_' + str(block_id) + '/depthwise_kernel:0'
    bias_name = 'conv_dw_' + str(block_id) + '/depthwise_bias:0'

    w, b, s = get_weights(weights, conv_name, bias_name, quant=False)
    x = depthwise_conv2d(x, w, b, strides=strides, padding='SAME')
    mean, std, beta, gamma = get_bn_param(weights, bn_mean, bn_var, bn_beta, bn_gamma)
    x = batch_norm(x, mean, std, beta, gamma)
    x = tf.nn.relu6(x)

    # pointwise convolution
    bn_beta = 'conv_pw_' + str(block_id) + '_bn/beta:0'
    bn_gamma = 'conv_pw_' + str(block_id) + '_bn/gamma:0'
    bn_mean = 'conv_pw_' + str(block_id) + '_bn/moving_mean:0'
    bn_var = 'conv_pw_' + str(block_id) + '_bn/moving_variance:0'
    conv_name = 'conv_pw_' + str(block_id) + '/kernel:0'
    bias_name = 'conv_pw_' + str(block_id) + '/bias:0'

    w, b, s = get_weights(weights, conv_name, bias_name)
    x = conv_2d(x, w, b, s, strides=1, padding='SAME')
    mean, std, beta, gamma = get_bn_param(weights, bn_mean, bn_var, bn_beta, bn_gamma)
    x = batch_norm(x, mean, std, beta, gamma)
    return tf.nn.relu6(x)


def MobileNet(img_input, weights, alpha):
    x = tf.reshape(img_input, shape=[-1, 224, 224, 3])

    # init convolution
    w, b, s = get_weights(weights, 'conv1/kernel:0', 'conv1/bias:0')
    x = conv_2d(x, w, b, s, strides=2, padding='SAME')
    mean, std, beta, gamma = get_bn_param(weights, 'conv1_bn/moving_mean:0',
                                          'conv1_bn/moving_variance:0', 'conv1_bn/beta:0', 'conv1_bn/gamma:0')
    x = batch_norm(x, mean, std, beta, gamma)
    x =tf.nn.relu6(x)

    x = _depthwise_conv_block(x, weights, block_id=1)

    x = _depthwise_conv_block(x, weights, strides=2, block_id=2)
    x = _depthwise_conv_block(x, weights, block_id=3)

    x = _depthwise_conv_block(x, weights,strides=2, block_id=4)
    x = _depthwise_conv_block(x, weights, block_id=5)

    x = _depthwise_conv_block(x, weights, strides=2, block_id=6)
    x = _depthwise_conv_block(x, weights, block_id=7)
    x = _depthwise_conv_block(x, weights, block_id=8)
    x = _depthwise_conv_block(x, weights, block_id=9)
    x = _depthwise_conv_block(x, weights, block_id=10)
    x = _depthwise_conv_block(x, weights, block_id=11)

    x = _depthwise_conv_block(x, weights, strides=2, block_id=12)
    x = _depthwise_conv_block(x, weights, block_id=13)

    x = avgpool_2d(x, k=7)

    x = tf.reshape(x, shape=[-1, 1, 1, int(1024 * alpha)])
    w, b, s = get_weights(weights, 'conv_preds/kernel:0', 'conv_preds/bias:0')
    x = conv_2d(x, w, b, s, strides=1, padding='SAME')
    x = tf.reshape(x, shape=[-1, 1000])
    return x
