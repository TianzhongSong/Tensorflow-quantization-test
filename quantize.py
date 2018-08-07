import argparse
import h5py
import numpy as np


def quantize(weights):
    abs_weights = np.abs(weights)
    vmax = np.max(abs_weights)
    s = vmax / 127.
    qweights = weights / s
    qweights = np.round(qweights)
    qweights = qweights.astype(np.int8)
    return qweights, s


def convert_weights(weight_file, output_weights):
    weights = h5py.File(weight_file, mode='r')
    qweights = h5py.File(output_weights, mode='w')
    try:
        layers = weights.attrs['layer_names']
    except:
        raise ValueError("weights file must contain attribution: 'layer_names'")
    qweights.attrs['layer_names'] = [name for name in weights.attrs['layer_names']]
    scales = []
    for layer_name in layers:
        f = qweights.create_group(layer_name)
        g = weights[layer_name]
        f.attrs['weight_names'] = g.attrs['weight_names']
        for weight_name in g.attrs['weight_names']:
            print(weight_name)
            weight_value = g[weight_name].value
            name = str(weight_name)
            name = name.split(':')[0]
            name = name.split('_')[-2]
            if name == 'W':
                new_weight_value, s = quantize(weight_value)
                scales.append(s)
                param_dest = f.create_dataset(weight_name, new_weight_value.shape, dtype=np.int8)
                param_dest[:] = new_weight_value
            else:
                param_dest = f.create_dataset(weight_name, weight_value.shape, dtype=np.float32)
                param_dest[:] = weight_value
    np.save('vgg_weights_scales.npy', np.array(scales))
    qweights.flush()
    qweights.close()
    print('Converting done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='command for converting keras pre-trained weights')
    parser.add_argument('--input-weights', type=str,
                        default='./weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    parser.add_argument('--output-weights', type=str, default='./weights/vgg16_tf_int8.h5')
    args = parser.parse_args()

    convert_weights(args.input_weights, args.output_weights)
