# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts Wider Face data to TFRecords file format with Example protos.
   
   Steps to create wider face dataset:
   1. Create top level directory called 'wider_dataset'.
   2. Extract WIDER_train.zip, WIDER_val.zip, WIDER_test.zip and wider_face_split.zip from
      http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
      (Face Training images, Face Validation images, Face Testing images and Face annotations)
      in wider_dataset.
   3. dataset_dir in run() function becomes top level directory ('wider_dataset' in this case)

"""
import os
import sys
import random
import bson
import cv2
import numpy as np
import scipy.ndimage
import tensorflow as tf
from   datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from   datasets.widerface_common import WIDERFACE_LABELS

# Original dataset organisation.
DIRECTORY_ANNOTATIONS  = 'wider_face_split/'
DIRECTORY_TRAIN_IMAGES = 'WIDER_train/images/'
DIRECTORY_VAL_IMAGES   = 'WIDER_val/images/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

# Populate 
def _populate_annotations_for_wider_database(bbox_file, dataset_info="Wider face dataset.", create_bson=False):
    def r_b(f_ptr):
        line_this = next(f_ptr).strip('\n')
        
        return line_this
    # enddef

    def r_fname_l(f_ptr):
        return r_b(f_ptr).split('--')

    def r_n_bbox(f_ptr):
        return int(r_b(f_ptr))

    def r_box_cord(f_ptr):
        c_l = [int(x) for x in r_b(f_ptr).split(' ')[0:-1]]
        #print c_l
        assert(len(c_l) == 10)
        return c_l
    # enddef

    bson_file = os.path.expandvars('$HOME') + '/' + os.path.basename(bbox_file) + '.bson'
    bbox_dict = {}
    bbox_list = []
    bbox_master_dict = {}

    with open(bbox_file, 'r') as in_file:
        try: 
            while True:
                img_this = r_b(in_file)
                n_bbox   = r_n_bbox(in_file)

                bbox_list_this = []
                bbox_dict[img_this] = []
                for box_this in range(0, n_bbox):
                    l_this = r_box_cord(in_file)
                    bbox_dict[img_this].append(l_this)
                    bbox_list_this.append(l_this)
                # endfor

                bbox_list.append({
                        "file" : img_this,
                        "bbox" : bbox_list_this,
                    })
            # endwhile
        except StopIteration as e:
            pass
        # endtry
    # endwith

    bbox_master_dict = {
                "data" : {
                             "dict" : bbox_dict,
                             "list" : bbox_list,
                         },
                "info" : dataset_info,
            }

    if create_bson:
        with open(bson_file, 'w') as f_out:
            f_out.write(bson.dumps(bbox_master_dict))
        # endwith

        return bbox_master_dict, bson_file
    # endif

    return bbox_master_dict
# endif

def _process_image(directory, img_dict):
    # Read the image file.
    filename = directory + DIRECTORY_TRAIN_IMAGES + img_dict['file']
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    bboxes     = img_dict['bbox']

    # Image shape.
    
    height, width, depth = scipy.ndimage.imread(filename).shape
    if height>1024:
        print (height)
    elif width>1024:
        print (width)
    shape = [ height, width, depth ]

    return image_data, shape, bboxes
# enddef


def _convert_to_example(image_data, bboxes, shape):
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    # shape
    height, width, channels = shape[0], shape[1], shape[2]
    # labels
    

    labels = []
    labels_text = []

    for b in bboxes:
        assert len(b) >= 4
        # Calculate absolute coordinates
        x1, y1 = b[0], b[1]
        x2, y2 = (b[0] + b[2]), (b[1] + b[3])
        # Normalize
        x1, x2 = float(x1)/width, float(x2)/width
        y1, y2 = float(y1)/height, float(y1)/height

        # Add labels
        label_this = 'face'
        labels.append(int(WIDERFACE_LABELS[label_this][0]))
        labels_text.append(label_this.encode('ascii'))

        # pylint: disable=expression-not-assigned

        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], [y1, x1, y2, x2])]
        # pylint: enable=expression-not-assigned
    # endfor

    # Debug
    if x1>=0.0 and x1<=1.0 and x2>=0.0 and x2<=1.0 and y1>=0.0 and y1<=1.0 and y2>=0.0 and y2<=1.0:

        image_format = b'JPEG'
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/channels': int64_feature(channels),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/label': int64_feature(labels),
            'image/object/label_text': bytes_feature(labels_text),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
        return example
    else:
        print x1,y1,x2,y2
        return None
    # endif
# enddef


def _add_to_tfrecord(dataset_dir, img_record, tfrecord_writer):
    image_data, shape, bboxes = _process_image(dataset_dir, img_record)
    example = _convert_to_example(image_data, bboxes, shape)
    if example is not None:
        tfrecord_writer.write(example.SerializeToString())
    # endif
# enddef


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='wider_train', shuffling=False):
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    # endif

    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    train_annot_path = os.path.join(path, 'wider_face_train_bbx_gt.txt')
    val_annot_path   = os.path.join(path, 'wider_face_val_bbx_gt.txt')

    # Check files
    if not tf.gfile.Exists(train_annot_path):
        print '{} does not exist. Exiting.'.format(train_annot_path)
        return
    # endif
    if not tf.gfile.Exists(val_annot_path):
        print '{} does not exist. Exiting.'.format(val_annot_path)
        return
    # endif

    # Get list of records
    #train_rec_list = bson.loads(open(train_annot_path, 'r').read())['data']['list']
    #val_rec_list   = bson.loads(open(val_annot_path, 'r').read())['data']['list']
    train_rec_list = _populate_annotations_for_wider_database(train_annot_path, dataset_info="Wider train dataset")['data']['list']
    val_rec_list   = _populate_annotations_for_wider_database(val_annot_path, dataset_info="Wider validation dataset")['data']['list']

    # shuffling
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(train_rec_list)
        random.shuffle(val_rec_list)
    # endif

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(train_rec_list):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(train_rec_list) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(train_rec_list)))
                sys.stdout.flush()

                _add_to_tfrecord(dataset_dir, train_rec_list[i], tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting Wider Face dataset!')
# endif
