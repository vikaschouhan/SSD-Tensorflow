#!/usr/bin/env python

import os
import math
import random
import numpy as np
import tensorflow as tf
import sys
from   nets import ssd_vgg_300, ssd_common, np_methods
from   preprocessing import ssd_vgg_preprocessing
import argparse
import matplotlib.image as mpimg

sys.path.append('.')
slim = tf.contrib.slim

# =========================================================================== #
# Print information
# =========================================================================== #
def list_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Listing bounding boxes. Largely inspired by SSD-MXNET!
    """
    height = img.shape[0]
    width = img.shape[1]
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            class_name = str(cls_id)

            print 'xmin={:04d}, ymin={:04d}, xmax={:04d}, ymax={:04d}, class_name={}'.format(xmin, ymin, xmax, ymax, class_name)
        # endif
    # endfor
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
        print "--image is required !!"
        sys.exit(-1)
    # endif
    if not args.__dict__["ckpt"]:
        print "--ckpt is required !!"
        sys.exit(-1)
    # endif

    # Test on some demo image and visualize output.
    img = mpimg.imread(args.__dict__["image"])
    rclasses, rscores, rbboxes =  process_image(img, args.__dict__["ckpt"])
   
    print 'Listing bounding box information ..'
    list_bboxes(img, rclasses, rscores, rbboxes)
# enddef

if __name__ == '__main__':
    main()
# endif
