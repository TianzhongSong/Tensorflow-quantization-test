# coding=utf8
from models.ssd300 import ssd_300
from models.ssd512 import ssd_512
from utils.object_detection_2d_data_generator import DataGenerator
from utils.average_precision_evaluator import Evaluator
from utils.coco_utils import get_coco_category_maps, predict_all_to_json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.load_weights import weight_loader
import tensorflow as tf
import argparse

weights = {'ssd300voc': 'VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5',
           'ssd512voc': 'VGG_VOC0712Plus_SSD_512x512_iter_240000.h5',
           'ssd300coco': 'VGG_coco_SSD_300x300_iter_400000.h5',
           'ssd512coco': 'VGG_coco_SSD_512x512_iter_360000.h5',
           'yolov3': 'yolov3.h5'}

device = '/cpu:0'


class SSD300():
    def __init__(self, weight_path=None, batch_szie=1, dataset='voc2007'):
        self.sess = tf.Session()
        self.X = tf.placeholder(tf.float32, [None, 300, 300, 3])
        self.Y = tf.placeholder(tf.float32, [None, 200, 6])
        self.weights = weight_loader(weight_path)
        self.batch_size = batch_szie
        self.predictions = self.generate(dataset)

    def generate(self, dataset='voc2007'):
        with tf.device(device):
            pred = ssd_300(self.X, self.weights,
                                 image_size=(300, 300, 3),
                                 n_classes=20 if dataset == 'voc2007' else 80,
                                 scales=[0.1, 0.2, 0.37, 0.54,
                                   0.71, 0.88, 1.05] if dataset == 'voc2007' else [0.07, 0.15, 0.33,
                                                                                   0.51, 0.69, 0.87, 1.05],
                                 aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                          [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                          [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                          [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                          [1.0, 2.0, 0.5],
                                                          [1.0, 2.0, 0.5]],
                                 two_boxes_for_ar1=True,
                                 steps=[8, 16, 32, 64, 100, 300],
                                 offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 clip_boxes=False,
                                 variances=[0.1, 0.1, 0.2, 0.2],
                                 normalize_coords=True,
                                 subtract_mean=[123, 117, 104],
                                 swap_channels=[2, 1, 0],
                                 confidence_thresh=0.01,
                                 iou_threshold=0.45,
                                 top_k=200,
                                 nms_max_output_size=400,
                                 batch_size=self.batch_size)
            return pred

    def predict(self, inputs):
        predictions = self.sess.run(self.predictions, feed_dict={self.X: inputs})
        return predictions


class SSD512():
    def __init__(self, weight_path=None, batch_szie=1, dataset='voc2007'):
        self.sess = tf.Session()
        self.X = tf.placeholder(tf.float32, [None, 512, 512, 3])
        self.Y = tf.placeholder(tf.float32, [None, 200, 6])
        self.weights = weight_loader(weight_path)
        self.batch_size = batch_szie
        self.predictions = self.generate(dataset)

    def generate(self, dataset='voc2007'):
        with tf.device(device):
            pred = ssd_512(self.X, self.weights,
                           image_size=(512, 512, 3),
                           n_classes=20 if dataset == 'voc2007' else 80,
                           scales=[0.07, 0.15, 0.3, 0.45,
                                   0.6, 0.75, 0.9, 1.05] if dataset == 'voc2007' else [0.04, 0.1, 0.26,
                                                                                       0.42, 0.58, 0.74, 0.9, 1.06],
                           aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5],
                                                    [1.0, 2.0, 0.5]],
                           two_boxes_for_ar1=True,
                           steps=[8, 16, 32, 64, 128, 256, 512],
                           offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                           clip_boxes=False,
                           variances=[0.1, 0.1, 0.2, 0.2],
                           normalize_coords=True,
                           subtract_mean=[123, 117, 104],
                           swap_channels=[2, 1, 0],
                           confidence_thresh=0.01,
                           iou_threshold=0.45,
                           top_k=200,
                           nms_max_output_size=400,
                           batch_size=self.batch_size)
            return pred

    def predict(self, inputs):
        predictions = self.sess.run(self.predictions, feed_dict={self.X: inputs})
        return predictions


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=str, default='ssd300', help='supports ssd300, ssd512, yolo320, yolo416, yolo608')
    parse.add_argument('--eval-dataset', type=str, default='voc2007', help='supports voc2007, coco')
    args = parse.parse_args()

    if args.model not in ['ssd300', 'ssd512', 'yolo320', 'yolo416', 'yolo608']:
        raise ValueError('Only supports ssd300, ssd512, yolo320, yolo416 and yolo608, but recieves {}'.format(args.model))

    if args.model in ['yolo320', 'yolo416', 'yolo608'] and args.eval_dataset == 'voc2007':
        raise ValueError('YOLO model is trained on COCO so that YOLO only can be evaluated with COCO!')

    if args.eval_dataset == 'voc2007':
        batch_szie = 8
        img_height = int(args.model[-3:])
        img_width = img_height
        n_classes = 20

        if args.model == 'ssd300':
            weight_path = './weights/' + weights['ssd300voc']
            model = SSD300(weight_path, batch_szie=batch_szie, dataset='voc2007')
        else:
            weight_path = './weights/' + weights['ssd512voc']
            model = SSD512(weight_path, batch_szie=batch_szie, dataset='voc2007')
        Pascal_VOC_dataset_images_dir = '../../datasets/VOCdevkit/VOC2007/JPEGImages/'
        Pascal_VOC_dataset_annotations_dir = '../../datasets/VOCdevkit/VOC2007/Annotations/'
        Pascal_VOC_dataset_image_set_filename = '../../datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

        classes = ['background',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
        dataset = DataGenerator()
        dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
                          image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
                          annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=False,
                          ret=False)
        evaluator = Evaluator(model=model,
                              n_classes=n_classes,
                              data_generator=dataset)

        results = evaluator(img_height=img_height,
                            img_width=img_width,
                            batch_size=batch_szie,
                            data_generator_mode='resize',
                            round_confidences=False,
                            matching_iou_threshold=0.5,
                            border_pixels='include',
                            sorting_algorithm='quicksort',
                            average_precision_mode='sample',
                            num_recall_points=11,
                            ignore_neutral_boxes=True,
                            return_precisions=True,
                            return_recalls=True,
                            return_average_precisions=True,
                            verbose=True)
        mean_average_precision, average_precisions, precisions, recalls = results
        print('Evaluating {0} with {1}'.format(args.model, args.eval_dataset))

        for i in range(1, len(average_precisions)):
            print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
        print()
        print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))

    elif args.eval_dataset == 'coco':
        batch_szie = 20
        img_height = int(args.model[-3:])
        img_width = img_height
        n_classes = 80

        if args.model == 'ssd300':
            weight_path = './weights/' + weights['ssd300coco']
            model = SSD300(weight_path, batch_szie=batch_szie, dataset='coco')
        else:
            weight_path = './weights/' + weights['ssd512coco']
            model = SSD512(weight_path, batch_szie=batch_szie, dataset='coco')

        dataset = DataGenerator()

        # Set the paths to the dataset here.
        MS_COCO_dataset_images_dir = '../../datasets/val2017/'
        MS_COCO_dataset_annotations_filename = '../../datasets/annotations/instances_val2017.json'

        dataset.parse_json(images_dirs=[MS_COCO_dataset_images_dir],
                           annotations_filenames=[MS_COCO_dataset_annotations_filename],
                           ground_truth_available=False,
                           include_classes='all',
                           ret=False)

        cats_to_classes, classes_to_cats, cats_to_names, classes_to_names = get_coco_category_maps(
            MS_COCO_dataset_annotations_filename)

        results_file = 'detections_val2017_ssd300_results.json'
        batch_size = 20
        predict_all_to_json(out_file=results_file,
                            model=model,
                            img_height=img_height,
                            img_width=img_width,
                            classes_to_cats=classes_to_cats,
                            data_generator=dataset,
                            batch_size=batch_size,
                            data_generator_mode='resize',
                            confidence_thresh=0.01,
                            iou_threshold=0.45,
                            top_k=200,
                            normalize_coords=True,
                            mode=args.model)
        coco_gt = COCO(MS_COCO_dataset_annotations_filename)
        coco_dt = coco_gt.loadRes(results_file)
        image_ids = sorted(coco_gt.getImgIds())

        cocoEval = COCOeval(cocoGt=coco_gt,
                            cocoDt=coco_dt,
                            iouType='bbox')
        cocoEval.params.imgIds = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    else:
        raise ValueError('Only support VOC2007 and COCO!')
