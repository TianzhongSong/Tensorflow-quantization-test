import numpy as np
import pickle as pkl
import cv2
import tqdm


class DataGenerator(object):
    def __init__(self, pkl_file, model='vgg', dtype='float32'):
        self.pkl_file = pkl_file
        self.model = model
        self.dtype = dtype

    def generator(self):
        data = self.load_pickle(self.pkl_file)
        assert len(data['data']) == 50000, len(data['data'])
        assert len(data['target']) == 50000, len(data['target'])
        for im, target in tqdm.tqdm(zip(data['data'], data['target']), total=50000):
        # for im, target in zip(data['data'], data['target']):
            im = self.str2img(im)
            if self.model not in ['inception', 'xception', 'mobilenet', 'inception_resnet']:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if self.model == 'squeezenet':
                im = cv2.resize(im, (227, 227))
            if self.model in ['inception', 'xception', 'inception_resnet']:
                im = cv2.resize(im, (299, 299))
            im = self.preprocessing(im, model=self.model)
            label = int(target)
            yield im, label


    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as f:
            v = pkl.load(f)
        f.close()
        return v

    @staticmethod
    def str2img(str_im):
        return cv2.imdecode(np.fromstring(str_im, np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def preprocessing(im, model='vgg', dtype='float32'):
        dtype = np.float16 if dtype == 'float16' else np.float32
        im = im.astype(dtype)
        if model in ['vgg', 'resnet', 'squeezenet']:
            im[..., 0] -= 103.939
            im[..., 1] -= 116.779
            im[..., 2] -= 123.68
        elif model in ['inception', 'mobilenet', 'xception', 'inception_resnet']:
            im /= 255.
            im -= 0.5
            im *= 2.
        elif model == 'densenet':
            im[..., 0] -= 103.939
            im[..., 1] -= 116.779
            im[..., 2] -= 123.68
            im[..., 0] *= 0.017
            im[..., 1] *= 0.017
            im[..., 2] *= 0.017
        else:
            pass
        return im
