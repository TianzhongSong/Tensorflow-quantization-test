# Tensorflow-Quantization-Test

CNN Quantization research.

### Requirements

keras==2.1.0

tensorflow==1.8.0

opencv==3.2.0

tqdm

h5py

### Quantization Method

Simply map |max| to 127. Reference from [TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf).

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

    python eval_image_classification.py --model='resnet'

An example for testing mobilenet with a width multiplier 1.0.

    python eval_image_classification.py --model='mobilenet' --alpha=1.0
    
### ImageNet Datatset

[ImageNet val data](http://ml.cs.tsinghua.edu.cn/~chenxi/dataset/val224_compressed.pkl) 
provided by [aaron-xichen](https://github.com/aaron-xichen), 
sincerely thanks to aaron-xichen for sharing this processed ImageNet val data.

### Results

Notice: MobileNets suffer significant accuracy loss.

<table width="95%">
  <tr>
    <td></td>
    <td colspan=2 align=center>float32</td>
    <td colspan=2 align=center>quantized</td>
    <td colspan=2 align=center>diff</td>
  </tr>
  <tr>
    <td align=center><b>Model</td>
    <td align=center>Top1 acc</td>
    <td align=center>Top5 acc</td>
    <td align=center>Top1 acc</td>
    <td align=center>Top5 acc</td>
    <td align=center>Top1 acc</td>
    <td align=center>Top5 acc</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5">VGG16</a></td>
    <td align=center width="10%"><b>0.70786</td>
    <td align=center width="10%"><b>0.89794</td>
    <td align=center width="10%"><b>0.7066</td>
    <td align=center width="10%"><b>0.89714</td>
    <td align=center width="10%"><b>-0.00126</td>
    <td align=center width="10%"><b>-0.0008</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5">ResNet50</a></td>
    <td align=center width="10%"><b>0.74366</td>
    <td align=center width="10%"><b>0.91806</td>
    <td align=center width="10%"><b>0.74004</td>
    <td align=center width="10%"><b>0.91574</td>
    <td align=center width="10%"><b>-0.00362</td>
    <td align=center width="10%"><b>-0.00232</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5">Inceptionv3</a></td>
    <td align=center width="10%"><b>0.76518</td>
    <td align=center width="10%"><b>0.92854</td>
    <td align=center width="10%"><b>0.75982</td>
    <td align=center width="10%"><b>0.92658</td>
    <td align=center width="10%"><b>-0.00536</td>
    <td align=center width="10%"><b>-0.00196</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5">Xception</a></td>
    <td align=center width="10%"><b>0.77446</td>
    <td align=center width="10%"><b>0.93618</td>
    <td align=center width="10%"><b>0.7672</td>
    <td align=center width="10%"><b>0.93204</td>
    <td align=center width="10%"><b>-0.00726</td>
    <td align=center width="10%"><b>-0.00414</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5">Squeezenet</a></td>
    <td align=center width="10%"><b>0.52294</td>
    <td align=center width="10%"><b>0.76312</td>
    <td align=center width="10%"><b>0.519</td>
    <td align=center width="10%"><b>0.76032</td>
    <td align=center width="10%"><b>-0.00394</td>
    <td align=center width="10%"><b>-0.0028</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf.h5">MobileNet-1-0</a></td>
    <td align=center width="10%"><b>0.69856</td>
    <td align=center width="10%"><b>0.89174</td>
    <td align=center width="10%"><b>0.64294</td>
    <td align=center width="10%"><b>0.85656</td>
    <td align=center width="10%"><b>-0.05562</td>
    <td align=center width="10%"><b>-0.03518</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_7_5_224_tf.h5">MobileNet-7-5</a></td>
    <td align=center width="10%"><b>0.67726</td>
    <td align=center width="10%"><b>0.87838</td>
    <td align=center width="10%"><b>0.6367</td>
    <td align=center width="10%"><b>0.84952</td>
    <td align=center width="10%"><b>-0.04056</td>
    <td align=center width="10%"><b>-0.02886</td>
    </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_5_0_224_tf.h5">MobileNet-5-0</a></td>
    <td align=center width="10%"><b>0.6352</td>
    <td align=center width="10%"><b>0.85006</td>
    <td align=center width="10%"><b>0.5723</td>
    <td align=center width="10%"><b>0.80522</td>
    <td align=center width="10%"><b>-0.0629</td>
    <td align=center width="10%"><b>-0.04484</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_2_5_224_tf.h5">MobileNet-2-5</a></td>
    <td align=center width="10%"><b>0.5134</td>
    <td align=center width="10%"><b>0.75546</td>
    <td align=center width="10%"><b>0.34848</td>
    <td align=center width="10%"><b>0.58956</td>
    <td align=center width="10%"><b>-0.16492</td>
    <td align=center width="10%"><b>-0.1659</td>
  </tr>
</table>

Only quantize pointwise convolution in MobileNet

<table width="95%">
  <tr>
    <td></td>
    <td colspan=2 align=center>float32</td>
    <td colspan=2 align=center>quantized</td>
    <td colspan=2 align=center>diff</td>
  </tr>
  <tr>
    <td align=center><b>Model</td>
    <td align=center>Top1 acc</td>
    <td align=center>Top5 acc</td>
    <td align=center>Top1 acc</td>
    <td align=center>Top5 acc</td>
    <td align=center>Top1 acc</td>
    <td align=center>Top5 acc</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf.h5">MobileNet-1-0</a></td>
    <td align=center width="10%"><b>0.69856</td>
    <td align=center width="10%"><b>0.89174</td>
    <td align=center width="10%"><b>0.65254</td>
    <td align=center width="10%"><b>0.86164</td>
    <td align=center width="10%"><b>-0.04602</td>
    <td align=center width="10%"><b>-0.0301</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_7_5_224_tf.h5">MobileNet-7-5</a></td>
    <td align=center width="10%"><b>0.67726</td>
    <td align=center width="10%"><b>0.87838</td>
    <td align=center width="10%"><b>0.64654</td>
    <td align=center width="10%"><b>0.85646</td>
    <td align=center width="10%"><b>-0.03072</td>
    <td align=center width="10%"><b>-0.02192</td>
    </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_5_0_224_tf.h5">MobileNet-5-0</a></td>
    <td align=center width="10%"><b>0.6352</td>
    <td align=center width="10%"><b>0.85006</td>
    <td align=center width="10%"><b>0.59438</td>
    <td align=center width="10%"><b>0.8217</td>
    <td align=center width="10%"><b>-0.04082</td>
    <td align=center width="10%"><b>-0.02836</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_2_5_224_tf.h5">MobileNet-2-5</a></td>
    <td align=center width="10%"><b>0.5134</td>
    <td align=center width="10%"><b>0.75546</td>
    <td align=center width="10%"><b>0.46506</td>
    <td align=center width="10%"><b>0.71176</td>
    <td align=center width="10%"><b>-0.04834</td>
    <td align=center width="10%"><b>-0.0437</td>
  </tr>
</table>

## Semantic Segmentation Part

something to do

## Object Detection Part

something to do

## Reference

something to do
