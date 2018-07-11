import tensorflow as tf


def quantize(x):
    abs_value = tf.abs(x)
    vmax = tf.reduce_max(abs_value)
    s = tf.divide(vmax, 127.)
    x = tf.floor_div(x, s)
    return x, s


def conv_2d(x, w, s, b=None, strides=1, padding='SAME', activation='relu'):
    x, sx = quantize(x)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
    x = tf.multiply(x, tf.multiply(sx, s))
    if b is not None:
        x = tf.nn.bias_add(x, b)
    if activation == 'relu':
        x = tf.nn.relu(x)
    return x


def denselayer(x, w, b, s, activation=None):
    x, sx = quantize(x)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.matmul(x, w)
    x = tf.multiply(x, tf.multiply(sx, s))
    x = tf.add(x, b)
    if activation == "relu":
        x = tf.nn.relu(x)
    return x


def maxpool_2d(x, k=2, padding='SAME'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding=padding)


def avgpool_2d(x, k=2, padding='SAME'):
    # AvgPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k,1],
                          padding=padding)
