import argparse
from models.vgg16 import VGG16
from models.resnet50 import ResNet50
from utils.load_weights import weight_loader
from pkl_reader import DataGenerator
import tensorflow as tf
import numpy as np


def top5_acc(pred, k=5):
    Inf = 0.
    results = []
    for i in range(k):
        results.append(pred.index(max(pred)))
        pred[pred.index(max(pred))] = Inf
    return results


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='command for testing keras model with fp16 and fp32')
    parse.add_argument('--model', type=str, default='resnet', help='support vgg16, resnet50, densenet121, \
         inceptionv3, inception_resnet, xception, mobilenet, squeezenet')
    parse.add_argument('--weights', type=str, default='./weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    parse.add_argument('--alpha', type=float, default=0.5, help='alpha for mobilenet')
    args = parse.parse_args()

    weights = weight_loader(args.weights, by_name=True)

    if args.model in ['vgg', 'resnet', 'inception_resnet', 'mobilenet']:
        X = tf.placeholder(tf.float32, [None, 224, 224, 3])
    elif args.model in ['inception', 'xception']:
        X = tf.placeholder(tf.float32, [None, 299, 299, 3])
    else:
        raise ValueError("Do not support {}".format(args.model))

    Y = tf.placeholder(tf.float32, [None, 1000])

    dg = DataGenerator('./data/val224_compressed.pkl', model=args.model, dtype='float32')
    with tf.device('/cpu:0'):
        logits, conv1 = ResNet50(X, weights)
        prediction = tf.nn.softmax(logits)
        pred = tf.argmax(prediction, 1)

    acc = 0.
    acc_top5 = 0.
    with tf.Session() as sess:
        for im, label in dg.generator():
            t1, t5, c1 = sess.run([pred, prediction, conv1], feed_dict={X: im})
            # print(t5)
            # print(t1[0])
            # print(c1)
            if t1[0] == label:
                acc += 1
            if label in top5_acc(t5[0].tolist()):
                acc_top5 += 1
            # print(t1[0], label)
        print('Top1 accuracy: {}'.format(acc / 50000))
        print('Top5 accuracy: {}'.format(acc_top5 / 50000))
