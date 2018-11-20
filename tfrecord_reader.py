import tensorflow as tf
import glob

def _parse(record):
    features = {
        # 'image/encoded': tf.FixedLenFeature([], tf.string),
        'data/encoded': tf.FixedLenFeature([], tf.string),
        'data/height': tf.FixedLenFeature([], tf.int64),
        'data/width': tf.FixedLenFeature([], tf.int64),
        'data/depth': tf.FixedLenFeature([], tf.int64),
        'data/class/label': tf.FixedLenFeature([], tf.int64),
        'data/class/name': tf.FixedLenFeature([], tf.string)

    }

    parsed_record = tf.parse_single_example(record, features)
    # image = tf.decode_raw(parsed_record['image/encoded'], tf.float32)
    image = tf.image.decode_image(parsed_record['data/encoded'])

    height = tf.cast(parsed_record['data/height'], tf.int32)
    width = tf.cast(parsed_record['data/width'], tf.int32)
    channel = tf.cast(parsed_record['data/depth'], tf.int32)

    image = tf.reshape(image, [height, width, channel])
    image = tf.image.resize_images(image, (299, 299))
    label = tf.cast(parsed_record['data/class/label'], tf.int32)
    lebel_name = tf.cast(parsed_record['data/class/name'], tf.string)

    return image, label, lebel_name

# train_input_fn = data_input_fn(glob.glob('H:/train_*.tfrecord'), shuffle=True)
# validation_input_fn = data_input_fn(glob.glob('H:/val_*.tfrecord'))


import matplotlib.pyplot as plt
with tf.Graph().as_default():
    with tf.Session() as sess:

        dataset = tf.data.TFRecordDataset(glob.glob('./train_*.tfrecord')).map(_parse)

        if True:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.repeat(None)  # Infinite iterations: let experiment determine num_epochs
        dataset = dataset.batch(64)

        iterator = dataset.make_one_shot_iterator()
        # iterator = dataset.make_initializable_iterator()
        features, labels, name = iterator.get_next()

        # sess.run(iterator)
        for i in range(10):
            d_l = sess.run(name)
            print(d_l[0])
            # print(d_n[0])
            # plt.imshow(d_f[0])
            # plt.show()
            # print(len(sess.run(features)))
