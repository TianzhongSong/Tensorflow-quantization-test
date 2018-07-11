from utils.layers import *
import tensorflow as tf

def VGG16(x, weights, scales):
    x = tf.reshape(x, shape=[-1, 224, 224, 3])

    # block 1
    w = tf.constant(weights['block1_conv1_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block1_conv1_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block1_conv1_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)

    w = tf.constant(weights['block1_conv2_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block1_conv2_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block1_conv2_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)
    x = maxpool_2d(x, k=2)

    # block2
    w = tf.constant(weights['block2_conv1_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block2_conv1_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block2_conv1_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)

    w = tf.constant(weights['block2_conv2_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block2_conv2_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block2_conv2_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)
    x = maxpool_2d(x, k=2)

    # block3
    w = tf.constant(weights['block3_conv1_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block3_conv1_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block3_conv1_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)

    w = tf.constant(weights['block3_conv2_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block3_conv2_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block3_conv2_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)

    w = tf.constant(weights['block3_conv3_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block3_conv3_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block3_conv3_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)
    x = maxpool_2d(x, k=2)

    # block4
    w = tf.constant(weights['block4_conv1_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block4_conv1_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block4_conv1_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)

    w = tf.constant(weights['block4_conv2_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block4_conv2_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block4_conv2_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)

    w = tf.constant(weights['block4_conv3_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block4_conv3_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block4_conv3_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)
    x = maxpool_2d(x, k=2)

    # block5
    w = tf.constant(weights['block5_conv1_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block5_conv1_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block5_conv1_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)

    w = tf.constant(weights['block5_conv2_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block5_conv2_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block5_conv2_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)

    w = tf.constant(weights['block5_conv3_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['block5_conv3_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['block5_conv3_W_1:0'], dtype=tf.float32)
    x = conv_2d(x, w, s, b)
    x = maxpool_2d(x, k=2)

    # fc1
    w = tf.constant(weights['fc1_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['fc1_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['fc1_W_1:0'], dtype=tf.float32)
    x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
    x = denselayer(x, w, b, s, activation='relu')

    # fc2
    w = tf.constant(weights['fc2_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['fc2_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['fc2_W_1:0'], dtype=tf.float32)
    x = denselayer(x, w, b, s, activation='relu')

    # fc3
    w = tf.constant(weights['predictions_W_1:0'], dtype=tf.float32)
    b = tf.constant(weights['predictions_b_1:0'], dtype=tf.float32)
    s = tf.constant(scales['predictions_W_1:0'], dtype=tf.float32)
    x = denselayer(x, w, b, s)
    return x
