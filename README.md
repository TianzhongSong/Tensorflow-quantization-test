# Tensorflow-Quantization-Test

CNN Quantization research.

### Requirements

keras==2.1.0

tensorflow==1.8.0

opencv==3.2.0

tqdm

h5py

### Quantization Method

Simply map |max| to 127.

![quantize method](https://github.com/TianzhongSong/Tensorflow-quantization-test/blob/master/figures/quantize.PNG)

Tensor Values = FP32 scale factor * int8 array.

Here is an example for convolution layer.

![quantized convolution](https://github.com/TianzhongSong/Tensorflow-quantization-test/blob/master/figures/quantized_conv.PNG)

Quantization detail can be found in [layer.py](https://github.com/TianzhongSong/Tensorflow-quantization-test/blob/master/utils/layers.py)

### Implementation

In this repository, all weight files are trained with Keras which are stored as HDF5 format. I parse these weight files with h5py then import them into TensorFlow models.

For example, import pre-trained weights into a convolution layer (first convolution layer of VGG16) built with tensorflow as follow

![workflow](https://github.com/TianzhongSong/Tensorflow-quantization-test/blob/master/figures/workflow.PNG)

I have built VGG16, ResNet50, InceptionV3, Xception, MobileNet, Squeezenet.
These models are tested successfully. For detail see [models directory](https://github.com/TianzhongSong/Tensorflow-quantization-test/tree/master/models)

## Image Classification Part

### Usage

An example for testing resnet50.

    python run_image_classification.py --model='resnet' --weights='resnet50_weights.h5'

### ImageNet Datatset

[ImageNet val data](http://ml.cs.tsinghua.edu.cn/~chenxi/dataset/val224_compressed.pkl) 
provided by [aaron-xichen](https://github.com/aaron-xichen), 
sincerely thanks to aaron-xichen for sharing this processed ImageNet val data.

### Results

|Model                  | float32              |float16                 |diff                  |
| :-------------------: |:--------------------:|:---------------------: |:-----------------------:|
|[VGG16](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5)                 | 0.70786/0.89794      | 0.7066/0.89714         | -0.00126/-0.0008   |
|[ResNet50](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5)               | 0.74366/0.91806      | 0.74004/0.91574        | -0.00362/-0.00232    |
|[Inceptionv3](https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5)            | 0.76518/0.92854      | 0.75982/0.92658          | -0.00536/0.00196    |
|[Xception](https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5)               | 0.77446/0.93618      | 0.7672/0.93204        |  -0.00726/-0.00414    |
|[Squeezenet](https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5)             | 0.52294/0.76312      | 0.519/0.76032        |   -0.00394/-0.0028     |
|[MobileNet-1-0](https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf.h5)          | 0.69856/0.89174      | 0.6966/0.8898          |   -0.00196/-0.00194    |
|[MobileNet-7-5](https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_7_5_224_tf.h5)          | 0.67726/0.87838      | 0.64654/0.85646         |   -0.03072/-0.02192    |
|[MobileNet-5-0](https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_5_0_224_tf.h5)          | 0.6352/0.85006       | 0.59438/0.8217        |   -0.04082/-0.02836   |
|[MobileNet-2-5](https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_2_5_224_tf.h5)          | 0.5134/0.75546       | 0.46506/0.71176         |  -0.04834/-0.0437   |

## Semantic Segmentation Part

something to do

## Object Detection Part

something to do

## Reference

something to do
