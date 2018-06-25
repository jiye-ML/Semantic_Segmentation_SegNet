import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl


class SegNet(object):

    def __init__(self, data, learning_rate, filter_size):
        # model
        self.filter_size = filter_size
        self.learning_rate = learning_rate
        # data
        self.data = data
        self.class_number = self.data.class_number
        self.image_height = self.data.image_height
        self.image_width = self.data.image_width
        self.image_channel = self.data.image_channel
        self.label_channel = self.data.label_channel
        self.label_channel = self.data.label_channel
        self.batch_size = self.data.batch_size

        self.graph = tf.Graph()

        self.images, self.labels = None, None
        self.x, self.y, self.logits, self.loss, self.train_op = None, None, None, None, None
        self.prediction, self.is_correct, self.accuracy = None, None, None

        with self.graph.as_default():
            # input
            self.x = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_channel], name="x")
            self.y = tf.placeholder(tf.uint8, [None, self.image_height, self.image_width, self.label_channel], name="y")
            # output
            self.logits = self.seg_net2(filter_size=self.filter_size)
            self.prediction = tf.reshape(tf.cast(tf.argmax(self.logits, axis=3), tf.uint8),
                                         shape=[-1, self.image_height, self.image_width, self.label_channel])  # 预测 输出
            self.is_correct = tf.cast(tf.equal(self.prediction, tf.cast(self.y, tf.uint8)), tf.uint8)  # 是否正确
            self.accuracy = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))  # 正确率

            # loss
            self.loss = self.loss_cross_entropy()
            # train
            self.train_op = self.get_train_op()

            # load batch
            self.images, self.labels = self.data.get_train_data()
            pass

        pass

    def seg_net2(self, filter_size):
        normal = tf.nn.lrn(self.x, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75, name="normalize")

        conv1 = self.conv_layer_with_bias_one(normal, [7, 7, self.image_channel, filter_size], name="conv1") # 360x480x16
        pool1, pool1_indices = self.conv_pool_layer_with_bias(conv1, [7, 7, filter_size, filter_size], count=1, name="pool1") # 180x240x16
        pool2, pool2_indices = self.conv_pool_layer_with_bias(pool1, [7, 7, filter_size, filter_size], count=2, name="pool2") # 90x120x16
        pool3, pool3_indices = self.conv_pool_layer_with_bias(pool2, [7, 7, filter_size, filter_size], count=3, name="pool3") # 45x60x16
        pool4, pool4_indices = self.conv_pool_layer_with_bias(pool3, [7, 7, filter_size, filter_size], count=3, name="pool4")  # 23x30x16
        pool5, pool5_indices = self.conv_pool_layer_with_bias(pool4, [7, 7, filter_size, filter_size], count=3, name="pool5") # 12x15x16

        up5 = self.deconv_layer(pool5, [2, 2, filter_size, filter_size], [self.batch_size, int(np.ceil(self.image_height / 16)), int(np.ceil(self.image_width / 16)), filter_size], name="up5")
        de5 = self.conv_layer_with_bias(up5, [7, 7, filter_size, filter_size], count=3, name="de5")
        up4 = self.deconv_layer(de5, [2, 2, filter_size, filter_size], [self.batch_size, self.image_height // 8, self.image_width // 8, filter_size], name="up4")
        de4 = self.conv_layer_with_bias(up4, [7, 7, filter_size, filter_size], count=3, name="de4")
        up3 = self.deconv_layer(de4, [2, 2, filter_size, filter_size], [self.batch_size, self.image_height // 4, self.image_width // 4, filter_size], name="up3")
        de3 = self.conv_layer_with_bias(up3, [7, 7, filter_size, filter_size], count=3, name="de3")
        up2 = self.deconv_layer(de3, [2, 2, filter_size, filter_size], [self.batch_size, self.image_height // 2, self.image_width // 2, filter_size], name="up2")
        de2 = self.conv_layer_with_bias(up2, [7, 7, filter_size, filter_size], count=2, name="de2")
        up1 = self.deconv_layer(de2, [2, 2, filter_size, filter_size], [self.batch_size, self.image_height, self.image_width, filter_size], name="up1")
        de1 = self.conv_layer_with_bias(up1, [7, 7, filter_size, filter_size], count=2, name="de1")

        with tf.variable_scope('conv'):
            kernel = tf.get_variable('weight', [1, 1, filter_size, self.data.class_number], initializer=self.msra_initializer(1, 64))
            conv = tf.nn.conv2d(de1, kernel, [1, 1, 1, 1], padding='SAME', name="conv")
            biases = self.get_bias_variable([32], name="conv_bias")
            logits = tf.nn.bias_add(conv, biases)
        return logits

    def seg_net(self, filter_size):
        normal = tf.nn.lrn(self.x, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75, name="normalize")

        pool1, pool1_indices = self.conv_pool_layer_with_bias(normal, [7, 7, self.image_channel, filter_size], count=1, name="pool1")
        pool2, pool2_indices = self.conv_pool_layer_with_bias(pool1, [7, 7, filter_size, filter_size], count=1, name="pool2")
        pool3, pool3_indices = self.conv_pool_layer_with_bias(pool2, [7, 7, filter_size, filter_size], count=1, name="pool3")
        pool4, pool4_indices = self.conv_pool_layer_with_bias(pool3, [7, 7, filter_size, filter_size], count=1, name="pool4")

        up4 = self.deconv_layer(pool4, [2, 2, filter_size, filter_size],
                                [self.batch_size, self.image_height // 8, self.image_width // 8, filter_size], name="up4")
        de4 = self.conv_layer_with_bias(up4, [7, 7, filter_size, filter_size], count=1, name="de4")
        up3 = self.deconv_layer(de4, [2, 2, filter_size, filter_size],
                                [self.batch_size, self.image_height // 4, self.image_width // 4, filter_size], name="up3")
        de3 = self.conv_layer_with_bias(up3, [7, 7, filter_size, filter_size], count=1, name="de3")
        up2 = self.deconv_layer(de3, [2, 2, filter_size, filter_size],
                                [self.batch_size, self.image_height // 2, self.image_width // 2, filter_size], name="up2")
        de2 = self.conv_layer_with_bias(up2, [7, 7, filter_size, filter_size], count=1, name="de2")
        up1 = self.deconv_layer(de2, [2, 2, filter_size, filter_size],
                                [self.batch_size, self.image_height, self.image_width, filter_size], name="up1")
        de1 = self.conv_layer_with_bias(up1, [7, 7, filter_size, filter_size], count=1, name="de1")

        with tf.variable_scope('conv'):
            kernel = tf.get_variable('weight', [1, 1, filter_size, 32], initializer=self.msra_initializer(1, 64))
            conv = tf.nn.conv2d(de1, kernel, [1, 1, 1, 1], padding='SAME', name="conv")
            biases = self.get_bias_variable([32], name="conv_bias")
            logits = tf.nn.bias_add(conv, biases)
        return logits

    def loss_cross_entropy(self):
        with tf.name_scope('loss'):
            # reshape label
            labels = tf.squeeze(tf.cast(self.y, tf.int32), axis=3)
            # compute the cross entropy of logits vs labels
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.logits)
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
        return cross_entropy_mean

    def get_train_op(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    @staticmethod
    def max_pool_2x2_argmax(x, name):
        with tf.variable_scope(name):
            # return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'), None

    def deconv_layer(self, input_t, f_shape, output_shape, stride=2, name=None):
        with tf.variable_scope(name):
            weights = self.get_deconv_filter(f_shape)
            return tf.nn.conv2d_transpose(input_t, weights, output_shape, strides=[1, stride, stride, 1])
        pass

    def conv_pool_layer_with_bias(self, input, shape, count, name=None):
        with tf.variable_scope(name):
            for index in range(count):
                input = self.conv_layer_with_bias_one(input=input, shape=shape, name="{}_{}".format(name, index))
            return self.max_pool_2x2_argmax(input, name=name + "_pool")
        pass

    def conv_layer_with_bias(self, input, shape, count, name):
        with tf.variable_scope(name):
            for index in range(count):
                input = self.conv_layer_with_bias_one(input=input, shape=shape, name="{}_{}".format(name, index))
            return input
        pass

    # 最小模块
    def conv_layer_with_bias_one(self, input, shape, name):
        with tf.variable_scope(name):
            input = tf.nn.conv2d(input, self.get_weight_variable(shape, name=name + "_filter"), [1, 1, 1, 1], padding='SAME')
            input = tf.nn.bias_add(input, self.get_bias_variable([shape[3]], name=name + "_bias"))
            return tf.nn.relu(tcl.batch_norm(input, center=False, scope=name + "_bn"))
        pass

    def get_deconv_filter(self, f_shape):
        f = math.ceil(f_shape[0] / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(f_shape[0]):
            for y in range(f_shape[1]):
                bilinear[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        # 如果核2x2,那么左上角的元素总是最大的
        return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)

    @staticmethod
    def get_weight_variable(shape, name=None):
        return tf.get_variable(initializer=tf.truncated_normal(shape, stddev=0.1), name=name)

    @staticmethod
    def get_bias_variable(shape, name=None):
        return tf.Variable(initial_value=tf.constant(0.0, shape=shape), name=name)

    @staticmethod
    def msra_initializer(kl, dl):
        # kl for kernel size, dl for filter number
        return tf.truncated_normal_initializer(stddev=math.sqrt(2. / (kl ** 2 * dl)))

    pass

