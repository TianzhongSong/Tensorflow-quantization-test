import numpy as np


def relu_func(x):
    x = (abs(x) + x) / 2.
    return x


def flatten(inputs):
    return inputs.reshape(inputs.shape[0]*inputs.shape[1]*inputs.shape[2])


def softmax_func(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def conv_2d(inputs, filters, bias, input_scale=0., weights_scale=0., strides=1):
    """
    2d convolution only forward
    """
    input_shape = inputs.shape
    pad = np.zeros((input_shape[0]+2, input_shape[0]+2, input_shape[2]), dtype=np.int32)
    input_shape = pad.shape
    pad[1:input_shape[0]-1,1:input_shape[1]-1,:] = inputs
    inputs = pad
    filter_shape = filters.shape
    if len(input_shape) == 2:
        input_shape = (input_shape[0], input_shape[1], 1)
    row, col, _ = input_shape
    output_shape = ((row-2)//strides, (col-2)//strides, filter_shape[-1])
    fp32_out = np.zeros(output_shape, dtype=np.float32)

    for k_idx in range(filter_shape[-1]):
        kfilter = filters[..., k_idx]
        for r_idx in range(row-2):
            for c_idx in range(col-2):
                conv_tmp = 0
                for f_idx in range(kfilter.shape[-1]):
                    tmp = np.array(inputs[r_idx:r_idx+filter_shape[0], c_idx:c_idx+filter_shape[1],f_idx]) * np.array(kfilter[:,:,f_idx])
                    # print(tmp)
                    conv_tmp += np.sum(tmp)
                fp32_out[r_idx, c_idx, k_idx] = conv_tmp
        fp32_out[...,k_idx] *= (input_scale * weights_scale)
        fp32_out[...,k_idx] += bias[k_idx]
    return fp32_out


def maxpooling(inputs, pool_size, strides):
    input_shape = inputs.shape
    if len(input_shape) == 2:
        input_shape = (input_shape[0], input_shape[1], 1)
    row, col, c = input_shape
    output_shape = (row//strides[0], col//strides[1], c)
    out = np.zeros(output_shape, dtype=np.float32)

    for k_idx in range(c):
        for r_idx in range(row):
            if r_idx % strides[0] == 0:
                for c_idx in range(col):
                    if c_idx % strides[1] == 0:
                        out[r_idx//2, c_idx//2, k_idx] = np.max(inputs[r_idx:r_idx+pool_size[0], c_idx:c_idx+pool_size[1], k_idx])
    return out


def denselayer(nb_neurals, inputs, weights, bias, input_scale, weights_scale):
    out = np.zeros(nb_neurals, dtype=np.float32)
    inputs = inputs.astype(np.int32)
    for i in range(nb_neurals):
        tmp = np.array(inputs) * np.array(weights[:, i])
        tmp = np.sum(tmp) * input_scale * weights_scale
        tmp = tmp + bias[i]
        out[i] = tmp
    return out
