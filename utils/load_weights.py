import h5py
import sys
import numpy as np

sys.path.append('../')


def weight_loader(weight_file):
    weights = {}
    f = h5py.File(weight_file, mode='r')
    # f = f['model_weights']
    try:
        layers = f.attrs['layer_names']
    except:
        raise ValueError("weights file must contain attribution: 'layer_names'")
    for layer_name in layers:
        g = f[layer_name]
        for weight_name in g.attrs['weight_names']:
            weight_value = g[weight_name].value
            name = str(weight_name).split("'")[1]
            weights[name] = weight_value
    return weights
