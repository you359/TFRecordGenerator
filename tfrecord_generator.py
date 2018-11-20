import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.image as mpimg
from multiprocessing.pool import ThreadPool


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
    output_filename = '%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)


# examples callback methods
def filenames_parser(data_dir):
    """
    filenames_parser
    
    :param data_dir: data directory
    :return: 
        filenames : list of file paths
        class_names : list of class names
    """

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
    """
    data_to_tfexample
    
    :param data: dictionary variable for storing data
                 related with data_processing(return value)
    :param class_name: class_name variable must be added to methods parameter
    :param class_id: class_id variable must be added to methods parameter
    :return: 
        tf.train.Example
    """
    return tf.train.Example(features=tf.train.Features(feature={
        'data/encoded': _bytes_feature(data['raw_data']),
        'data/height': _int64_feature(data['height']),
        'data/width': _int64_feature(data['width']),
        'data/depth': _int64_feature(data['depth']),
        'data/class/label': _int64_feature(class_id),
        'data/class/name': _bytes_feature(tf.compat.as_bytes(class_name))
    }))


def data_processing(sess, filenames):
    """
    data_processing
    
    :param sess: sess variable must be added to methods parameter
                 it used for some tf based methods
    :param filenames: list of filenames
    :return: 
        data : dictionary variable for storing data
               related with data_to_tfexample(data, ,)  
    """

    data = {'raw_data':None, 'height':None, 'width':None, 'channel':None}
    data['raw_data'] = tf.gfile.FastGFile(filenames, 'rb').read()
    data['height'], data['width'], data['depth'] = mpimg.imread(filenames).shape
    return data

class TFRecordGenerator:
    def __init__(self,
                 data_dir,
                 output_dir,
                 filenames_parser,
                 data_to_tfexample,
                 data_processing,
                 split_names,
                 split_ratios):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.filenames_parser = filenames_parser

        if len(split_names) != len(split_ratios):
            raise ValueError('length of split_names must be the equal to length of split_ratios')

        if sum(split_ratios) != 1:
            raise ValueError('sum of all split_ratios must be 1')

        self.split_names = split_names
        self.split_ratio = split_ratios

        self.data_to_tfexample = data_to_tfexample
        self.data_processing = data_processing

    def convert_to(self, workers=1, num_shards=1):
        """
        convert_to
        
        :param workers: workers for ThreadPool (multiprocessing) 
        :param num_shards: number of shard
        :return: 
        """

        filenames, class_names = self.filenames_parser(self.data_dir)
        # Refer each of the class name to a specific integer number for predictions later
        class_names_to_ids = dict(zip(class_names, range(len(class_names))))

        # Find the number of validation examples we need
        num_splits = [[] for k in range(len(self.split_names))]

        for i in range(len(self.split_names)):
            num_splits[i] = int(self.split_ratio[i] * len(filenames))

        # Divide the training datasets into train and test:
        random.seed(0)
        random.shuffle(filenames)

        split_filenames = [[] for k in range(len(self.split_names))]
        prev_idx = 0
        for idx, num_split in enumerate(num_splits):
            next_idx = prev_idx + num_split
            split_filenames[idx] = filenames[prev_idx:next_idx]
            prev_idx = next_idx

        def _process_examples(
                sess,
                data_processing,
                data_to_tfexample,
                split_name,
                filenames,
                class_names_to_ids,
                shard_id,
                num_shards):

            if sess is None:
                with tf.Graph().as_default():
                    with tf.Session('') as sess:
                        output_filename = _get_dataset_filename(self.output_dir, split_name, shard_id, num_shards)

                        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                            pbar = tqdm(range(len(filenames)))
                            for i in pbar:
                                data = data_processing(sess, filenames[i])

                                class_name = os.path.basename(os.path.dirname(filenames[i]))
                                class_id = class_names_to_ids[class_name]

                                example = data_to_tfexample(
                                    data, class_name, class_id
                                )

                                tfrecord_writer.write(example.SerializeToString())
            else:
                output_filename = _get_dataset_filename(self.output_dir, split_name, shard_id, num_shards)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    pbar = tqdm(range(len(filenames)))
                    for i in pbar:
                        data = data_processing(sess, filenames[i])

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = data_to_tfexample(
                            data, class_name, class_id
                        )

                        tfrecord_writer.write(example.SerializeToString())

        if workers == 1:
            with tf.Graph().as_default():
                with tf.Session('') as sess:
                    if num_shards < workers:
                        raise ValueError('workers must be less than num_shards')

                    if num_shards == 1:
                        for i in range(len(self.split_names)):
                            _process_examples(sess,
                                              data_processing=self.data_processing,
                                              data_to_tfexample=self.data_to_tfexample,
                                              split_name=self.split_names[i],
                                              filenames=split_filenames[i],
                                              class_names_to_ids=class_names_to_ids,
                                              shard_id=1,
                                              num_shards=num_shards)
                    else:
                        for i in range(len(self.split_names)):
                            sharded_dataset = np.array_split(split_filenames[i], num_shards)

                            for shard, dataset in enumerate(sharded_dataset):
                                _process_examples(sess,
                                                  data_processing=self.data_processing,
                                                  data_to_tfexample=self.data_to_tfexample,
                                                  split_name=self.split_names[i],
                                                  filenames=dataset,
                                                  class_names_to_ids=class_names_to_ids,
                                                  shard_id=shard,
                                                  num_shards=num_shards)
        else:
            pool = ThreadPool(processes=workers)
            for i in range(len(self.split_names)):
                sharded_dataset = np.array_split(split_filenames[i], num_shards)
                for shard, dataset in enumerate(sharded_dataset):
                    pool.apply_async(_process_examples,
                                     args=(
                                         None,
                                         self.data_processing,
                                         self.data_to_tfexample,
                                         self.split_names[i],
                                         dataset,
                                         class_names_to_ids,
                                         shard,
                                         num_shards
                                     ))
            pool.close()
            pool.join()
