import tensorflow as tf


def quantize(x):
    abs_value = tf.abs(x)
    vmax = tf.reduce_max(abs_value)
    s = tf.divide(vmax, 127.)
    x = tf.divide(x, s)
    x = tf.round(x)
    return x, s


def batch_norm(x, mean, variance, offset=None, scale=None):
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=1e-3)


def conv_2d(x, w, b=None, strides=1, padding='SAME', activation=''):
    '''
    2D convolution with quantization (float32-->int8)
    '''
    x, sx = quantize(x)
    w, sw = quantize(w)
    # Actually, convolution using float32, because of tensorflow does not support int8 conv op
    x = tf.cast(x, dtype=tf.float32)
    w = tf.cast(w, dtype=tf.float32)
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
    # multiply scales
    x = tf.multiply(x, tf.multiply(sx, sw))
    if b is not None:
        x = tf.nn.bias_add(x, b)
    if activation == 'relu':
        x = tf.nn.relu(x)
    return x


def depthwise_conv2d(x, w, b=None, strides=1, padding='SAME', activation=''):
    x, sx = quantize(x)
    w, sw = quantize(w)
    x = tf.cast(x, dtype=tf.float32)
    w = tf.cast(w, dtype=tf.float32)
    x = tf.nn.depthwise_conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
    # multiply scales
    x = tf.multiply(x, tf.multiply(sx, sw))
    if b is not None:
        x = tf.nn.bias_add(x, b)
    if activation == 'relu':
        x = tf.nn.relu(x)
    return x


def denselayer(x, w, b, activation=''):
    x, sx = quantize(x)
    w, sw = quantize(w)
    x = tf.cast(x, dtype=tf.float32)
    w = tf.cast(w, dtype=tf.float32)
    x = tf.matmul(x, w)
    x = tf.multiply(x, tf.multiply(sx, sw))
    x = tf.add(x, b)
    if activation == "relu":
        x = tf.nn.relu(x)
    return x


def maxpool_2d(x, k=2, s=2, padding='SAME'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding=padding)


def avgpool_2d(x, k=2, s=1, padding='VALID'):
    # AvgPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s,1],
                          padding=padding)
