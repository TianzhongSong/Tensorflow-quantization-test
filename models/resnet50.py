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


def get_bn_param(weights, mean, std, beta, gamma):
    mean = tf.constant(weights[mean], dtype=tf.float32)
    std = tf.constant(weights[std], dtype=tf.float32)
    beta = tf.constant(weights[beta], dtype=tf.float32)
    gamma = tf.constant(weights[gamma], dtype=tf.float32)
    return mean, std, beta, gamma


def identity_block(inputs, weights, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_params = ['_running_mean:0', '_running_std:0', '_beta:0', '_gamma:0']
    conv_wb = ['_W:0', '_b:0']
    conv_names = ['2a', '2b', '2c']

    conv = conv_name_base + conv_names[0]
    w, b, s = get_weights_biases_scale(weights,
                                    conv + conv_wb[0], conv + conv_wb[1])
    x = conv_2d(inputs, w, b, s)
    bn = bn_name_base + conv_names[0]
    mean, std, beta, gamma = get_bn_param(weights, bn + bn_params[0],
                                          bn + bn_params[1], bn + bn_params[2], bn + bn_params[3])
    x = batch_norm(x, mean, std, beta, gamma)
    x = tf.nn.relu(x)

    for i in range(1, 3):
        conv = conv_name_base + conv_names[i]
        w, b, s = get_weights_biases_scale(weights,
                                           conv + conv_wb[0], conv + conv_wb[1])
        x = conv_2d(x, w, b, s)
        bn = bn_name_base + conv_names[i]
        mean, std, beta, gamma = get_bn_param(weights, bn + bn_params[0],
                                              bn + bn_params[1], bn + bn_params[2], bn + bn_params[3])
        x = batch_norm(x, mean, std, beta, gamma)
        if i < 2:
            x = tf.nn.relu(x)
    x = tf.add(x, inputs)
    return tf.nn.relu(x)


def conv_block(inputs, weights, stage, block, strides=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_params = ['_running_mean:0', '_running_std:0', '_beta:0', '_gamma:0']
    conv_wb = ['_W:0', '_b:0']
    conv_names = ['2a', '2b', '2c']

    conv = conv_name_base + conv_names[0]
    w, b, s = get_weights_biases_scale(weights,
                                    conv + conv_wb[0], conv + conv_wb[1])
    x = conv_2d(inputs, w, b, s, strides=strides)
    bn = bn_name_base + conv_names[0]
    mean, std, beta, gamma = get_bn_param(weights, bn + bn_params[0],
                                          bn + bn_params[1], bn + bn_params[2], bn + bn_params[3])
    x = batch_norm(x, mean, std, beta, gamma)
    x = tf.nn.relu(x)

    for i in range(1, 3):
        conv = conv_name_base + conv_names[i]
        w, b, s = get_weights_biases_scale(weights,
                                           conv + conv_wb[0], conv + conv_wb[1])
        x = conv_2d(x, w, b, s)
        bn = bn_name_base + conv_names[i]
        mean, std, beta, gamma = get_bn_param(weights, bn + bn_params[0],
                                              bn + bn_params[1], bn + bn_params[2], bn + bn_params[3])
        x = batch_norm(x, mean, std, beta, gamma)
        if i < 2:
            x = tf.nn.relu(x)

    # shortcut
    w, b, s = get_weights_biases_scale(weights,
                                       conv_name_base + '1_W:0', conv_name_base + '1_b:0')
    shortcut = conv_2d(inputs, w, b, s, strides=strides)
    bn = bn_name_base + '1'
    mean, std, beta, gamma = get_bn_param(weights, bn + bn_params[0],
                                          bn + bn_params[1], bn + bn_params[2], bn + bn_params[3])
    shortcut = batch_norm(shortcut, mean, std, beta, gamma)
    x = tf.add(x, shortcut)
    return tf.nn.relu(x)


def ResNet50(x, weights):
    # init convolution
    x = tf.reshape(x, shape=[-1, 224, 224, 3])
    w, b, s = get_weights_biases_scale(weights, 'conv1_W:0', 'conv1_b:0')
    x = conv_2d(x, w, b, s, strides=2)
    mean, std, beta, gamma = get_bn_param(weights, 'bn_conv1_running_mean:0',
                                          'bn_conv1_running_std:0', 'bn_conv1_beta:0', 'bn_conv1_gamma:0')
    x = batch_norm(x, mean, std, beta, gamma)
    x = tf.nn.relu(x)
    x = maxpool_2d(x, k=3, s=2, padding='SAME')

    x = conv_block(x, weights, stage=2, block='a', strides=1)
    x1 = identity_block(x, weights, stage=2, block='b')
    x = identity_block(x1, weights, stage=2, block='c')

    x = conv_block(x, weights, stage=3, block='a')
    x = identity_block(x, weights, stage=3, block='b')
    x = identity_block(x, weights, stage=3, block='c')
    x = identity_block(x, weights, stage=3, block='d')

    x = conv_block(x, weights, stage=4, block='a')
    x = identity_block(x, weights, stage=4, block='b')
    x = identity_block(x, weights, stage=4, block='c')
    x = identity_block(x, weights, stage=4, block='d')
    x = identity_block(x, weights, stage=4, block='e')
    x = identity_block(x, weights, stage=4, block='f')

    x = conv_block(x, weights, stage=5, block='a')
    x = identity_block(x, weights, stage=5, block='b')
    x = identity_block(x, weights, stage=5, block='c')

    x = avgpool_2d(x, k=7)

    w, b, s = get_weights_biases_scale(weights, 'fc1000_W:0', 'fc1000_b:0')
    x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
    x = denselayer(x, w, b, s)
    return x
