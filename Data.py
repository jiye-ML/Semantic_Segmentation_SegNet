import os
import numpy as np
from PIL import Image
from tensorflow.python.framework import ops, dtypes
import tensorflow as tf

class CamVidData(object):

    def __init__(self, class_number, batch_size, data_root_path):
        # 数据形式
        self.image_height = 360
        self.image_width = 480
        self.image_channel = 3
        self.label_channel = 1
        self.class_number = class_number
        self.batch_size = batch_size
        # 数据目录
        self._train_data_dir = os.path.join(data_root_path, "train")
        self._train_anno_dir = os.path.join(data_root_path, "trainannot")
        self._test_data_dir = os.path.join(data_root_path, "test")
        self._test_anno_dir = os.path.join(data_root_path, "testannot")

        self.train_images_path = [os.path.join(self._train_data_dir, f) for f in os.listdir(self._train_data_dir)]
        self.train_annots_path = [os.path.join(self._train_anno_dir, f) for f in os.listdir(self._train_anno_dir)]
        self.test_images_path = [os.path.join(self._test_data_dir, f) for f in os.listdir(self._test_data_dir)]
        self.test_annots_path = [os.path.join(self._test_anno_dir, f) for f in os.listdir(self._test_anno_dir)]

        self.test_number = len(self.test_images_path)
        self.test_batch_number = self.test_number // self.batch_size
        pass

    def get_train_data(self):
        return self._get_train_data(self.train_images_path, self.train_annots_path)

    def get_test_data(self, i):
        i = 0 if i >= self.test_batch_number else i
        start = i * self.batch_size
        end = start + self.batch_size
        images = self.test_images_path[start: end]
        labels = self.test_annots_path[start: end]
        images_data = [np.copy(np.asarray(Image.open(image))) for image in images]
        labels_data = [np.copy(np.asarray(Image.open(label))) for label in labels]
        return images_data, np.reshape(labels_data, newshape=[len(labels_data), self.image_height,
                                                              self.image_width, self.label_channel])

    def _get_train_data(self, images_path, annots_path, shuffle=True):
        # 利用tensorflow形式处理
        image_path = ops.convert_to_tensor(images_path, dtype=dtypes.string)
        label_path = ops.convert_to_tensor(annots_path, dtype=dtypes.string)

        queue = tf.train.slice_input_producer([image_path, label_path], shuffle=shuffle)
        images, labels = self._reader_data(queue)
        images = tf.cast(images, tf.float32)
        capacity = int(0.4 * 367) + 3 * self.batch_size
        if shuffle:
            images, labels = tf.train.shuffle_batch([images, labels], batch_size=self.batch_size, num_threads=30,
                                                    capacity=capacity, min_after_dequeue=int(0.4 * 367))
        else:
            images, labels = tf.train.batch([images, labels], batch_size=self.batch_size, num_threads=30,
                                            capacity=capacity)
        return images, labels

    # 利用tensorflow队列，这样利用了多线程
    def _reader_data(self, queue):
        # file names
        image_name = queue[0]
        label_name = queue[1]
        # read bytes into tensor and convert from png format
        image_val = tf.image.decode_png(tf.read_file(image_name))
        label_val = tf.image.decode_png(tf.read_file(label_name))
        # reshape to image/label shape
        images = tf.reshape(image_val, (self.image_height, self.image_width, self.image_channel))
        labels = tf.reshape(label_val, (self.image_height, self.image_width, self.label_channel))
        return images, labels

    pass