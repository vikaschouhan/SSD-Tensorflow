# ==============================================================================
"""Provides data for the WiderFace Dataset (images + annotations).
"""
import tensorflow as tf
from datasets import widerface_common

slim = tf.contrib.slim

FILE_PATTERN = 'wider_%s_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}
SPLITS_TO_SIZES = {
    'train': 5011,
    'test': 4952,
    'val' : 20,
}
NUM_CLASSES = 2


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading WiderFace.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if file_pattern == None:
        file_pattern = FILE_PATTERN
    # endif
    return widerface_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES)
# enddef
