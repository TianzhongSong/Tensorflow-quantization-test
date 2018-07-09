import argparse
from pkl_reader import DataGenerator
from utils.layers import *
import numpy as np
import h5py


def top5_acc(pred, k=5):
    Inf = 0.
    results = []
    for i in range(k):
        results.append(pred.index(max(pred)))
        pred[pred.index(max(pred))] = Inf
    return results


def quantize(weights):
    abs_weights = np.abs(weights)
    vmax = np.max(abs_weights)
    s = vmax / 127.
    qweights = weights / s
    qweights = np.round(qweights)
    qweights = qweights.astype(np.int8)
    return qweights, s


if __name__ == "__main__":
    weights = []
    weight = h5py.File('./weights/vgg16_tf_int8.h5', mode='r')
    try:
        layers = weight.attrs['layer_names']
    except:
        raise ValueError("weights file must contain attribution: 'layer_names'")
    for layer_name in layers:
        g = weight[layer_name]
        for weight_name in g.attrs['weight_names']:
            print(weight_name)
            weight_value = g[weight_name].value
            weights.append(weight_value)
    weight.close()

    scales = np.load('vgg_weights_scales.npy')
    dg = DataGenerator('./data/val224_compressed.pkl', model='vgg', dtype='float32')
    acc = 0
    acc_top5 = 0
    for im, label in dg.generator():
        # block1
        x, si = quantize(im)
        x = conv_2d(x, filters=weights[0], bias=weights[1], input_scale=si, weights_scale=scales[0])
        x = relu_func(x)
        x, si = quantize(x)

        x = conv_2d(x, filters=weights[2], bias=weights[3], input_scale=si, weights_scale=scales[1])
        x = relu_func(x)
        x, si = quantize(x)
        x = maxpooling(x, pool_size=(2,2), strides=(2, 2))

        # block2
        x = conv_2d(x, filters=weights[4], bias=weights[5], input_scale=si, weights_scale=scales[2])
        x = relu_func(x)
        x, si = quantize(x)

        x = conv_2d(x, filters=weights[6], bias=weights[7], input_scale=si, weights_scale=scales[3])
        x = relu_func(x)
        x, si = quantize(x)
        x = maxpooling(x, pool_size=(2, 2), strides=(2, 2))

        # block3
        x = conv_2d(x, filters=weights[8], bias=weights[9], input_scale=si, weights_scale=scales[4])
        x = relu_func(x)
        x, si = quantize(x)

        x = conv_2d(x, filters=weights[10], bias=weights[11], input_scale=si, weights_scale=scales[5])
        x = relu_func(x)
        x, si = quantize(x)

        x = conv_2d(x, filters=weights[12], bias=weights[13], input_scale=si, weights_scale=scales[6])
        x = relu_func(x)
        x, si = quantize(x)
        x = maxpooling(x, pool_size=(2, 2), strides=(2, 2))

        # block4
        x = conv_2d(x, filters=weights[14], bias=weights[15], input_scale=si, weights_scale=scales[7])
        x = relu_func(x)
        x, si = quantize(x)

        x = conv_2d(x, filters=weights[16], bias=weights[17], input_scale=si, weights_scale=scales[8])
        x = relu_func(x)
        x, si = quantize(x)

        x = conv_2d(x, filters=weights[18], bias=weights[19], input_scale=si, weights_scale=scales[9])
        x = relu_func(x)
        x, si = quantize(x)
        x = maxpooling(x, pool_size=(2, 2), strides=(2, 2))

        # block5
        x = conv_2d(x, filters=weights[20], bias=weights[21], input_scale=si, weights_scale=scales[10])
        x = relu_func(x)
        x, si = quantize(x)

        x = conv_2d(x, filters=weights[22], bias=weights[23], input_scale=si, weights_scale=scales[11])
        x = relu_func(x)
        x, si = quantize(x)

        x = conv_2d(x, filters=weights[24], bias=weights[25], input_scale=si, weights_scale=scales[12])
        x = relu_func(x)
        x, si = quantize(x)
        x = maxpooling(x, pool_size=(2, 2), strides=(2, 2))

        x = flatten(x)

        x = denselayer(4096, x, weights[26], weights[27], input_scale=si, weights_scale=scales[13])
        x = relu_func(x)
        x, si = quantize(x)

        x = denselayer(4096, x, weights[28], weights[29], input_scale=si, weights_scale=scales[14])
        x = relu_func(x)
        x, si = quantize(x)

        x = denselayer(1000, x, weights[30], weights[31], input_scale=si, weights_scale=scales[15])
        x = softmax_func(x)

        pred = np.argmax(x[0])
        print('pred:{0},true:{1}'.format(pred, label))
        if pred == label:
            acc += 1
        if label in top5_acc(x[0].tolist()):
            acc_top5 += 1
    print('Top1 accuracy: {}'.format(acc / 1000))
    print('Top5 accuracy: {}'.format(acc_top5 / 1000))
