#!/usr/bin/env python

import os
import math
import random
import numpy as np
import tensorflow as tf
import cv2
import sys
from   nets import ssd_vgg_300, ssd_common, np_methods
from   preprocessing import ssd_vgg_preprocessing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm
import argparse

sys.path.append('.')
slim = tf.contrib.slim

## VISUALIZATION
# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors
# enddef

colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


# =========================================================================== #
# OpenCV drawing.
# =========================================================================== #
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw a collection of lines on an image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        # endfor
    # endfor
# enddef


def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
# enddef

def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
# enddef


def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % (classes[i], scores[i])
        p1 = (p1[0]-5, p1[1])
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)
    # endfor
# enddef


# =========================================================================== #
# Matplotlib show...
# =========================================================================== #
def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()
# enddef

## IMAGE PROCESSING

def process_image(img, ckpt_filename, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)

    # ## SSD 300 Model
    # 
    # The SSD 300 network takes 300x300 image inputs. In order to feed any image,
    # the latter is resize to this input shape (i.e.`Resize.WARP_RESIZE`). Note that
    # even though it may change the ratio width / height, the SSD model performs well on
    # resized images (and it is the default behaviour in the original Caffe implementation).
    # 
    # SSD anchors correspond to the default bounding boxes encoded in the network. The SSD
    # net output provides offset on the coordinates and dimensions of these anchors.
    
    # Input placeholder.
    net_shape = (300, 300)
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)
    
    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    ssd_net = ssd_vgg_300.SSDNet()
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
    
    # Restore SSD model.
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)
    
    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)

    # ## Post-processing pipeline
    # 
    # The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:
    # 
    # * Select boxes above a classification threshold;
    # * Clip boxes to the image shape;
    # * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
    # * If necessary, resize bounding boxes to original image shape.
    
    # Main image processing routine.
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes
# enddef


def main():
    parser  = argparse.ArgumentParser()
    parser.add_argument("--image", help="JPEG Image", type=str, default=None)
    parser.add_argument("--ckpt", help="model checkpoint file", type=str, default=None)
    args    = parser.parse_args()

    if not args.__dict__["image"]:
        print("--image is required !!")
        sys.exit(-1)
    # endif
    if not args.__dict__["ckpt"]:
        print("--ckpt is required !!")
        sys.exit(-1)
    # endif

    # Test on some demo image and visualize output.
    img = mpimg.imread(args.__dict__["image"])
    rclasses, rscores, rbboxes =  process_image(img, args.__dict__["ckpt"])
    
    plt_bboxes(img, rclasses, rscores, rbboxes)
# enddef

if __name__ == '__main__':
    main()
# endif
