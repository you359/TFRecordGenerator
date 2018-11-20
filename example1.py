"""
convert flower_photos dataset to tfrecords
"""

import os
import tensorflow as tf
from tfrecord_generator import _bytes_feature, _int64_feature, TFRecordGenerator
import matplotlib.image as mpimg

def filenames_parser(data_dir):
    flower_root = os.path.join(data_dir, 'flower_photos')
    directories = []
    class_names = []
    for filename in os.listdir(flower_root):
        path = os.path.join(flower_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            filenames.append(path)

    return filenames, class_names


def data_to_tfexample(data, class_name, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'data/encoded': _bytes_feature(data['raw_data']),
        'data/height': _int64_feature(data['height']),
        'data/width': _int64_feature(data['width']),
        'data/depth': _int64_feature(data['depth']),
        'data/class/label': _int64_feature(class_id),
        'data/class/name': _bytes_feature(tf.compat.as_bytes(class_name))
    }))


def data_processing(sess, filenames):
    data = {'raw_data':None, 'height':None, 'width':None, 'channel':None}
    data['raw_data'] = tf.gfile.FastGFile(filenames, 'rb').read()
    data['height'], data['width'], data['depth'] = mpimg.imread(filenames).shape
    return data

if __name__ == '__main__':
    tfrecord_generator = TFRecordGenerator(data_dir='H:/',
                                           output_dir='./',
                                           filenames_parser=filenames_parser,
                                           data_to_tfexample=data_to_tfexample,
                                           data_processing=data_processing,
                                           split_names=['train', 'val'],
                                           split_ratios=[0.7, 0.3])

    tfrecord_generator.convert_to(num_shards=5, workers=3)
