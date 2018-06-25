import tensorflow as tf
import os
import numpy as np
from PIL import Image

from Data import CamVidData
from Tools import Tools
from DataConfig import Config
from SegNet import SegNet

class Runner(object):

    def __init__(self, data, net, model_path, result_path):
        self.data = data
        self.net = net
        self.model_path = model_path
        self.result_path = result_path
        # 管理网络中创建的图
        self.supervisor = tf.train.Supervisor(graph=self.net.graph, logdir=self.model_path)
        self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        pass

    def train(self, epochs, loss_freq, model_freq, test_freq):
        with self.supervisor.managed_session(config=self.config) as sess:
            for step in range(epochs):
                image_b, label_b = sess.run([self.net.images, self.net.labels])
                ind = np.where(label_b == 11)
                label_b[ind] = 0
                image_b[:, ...] -= Config[self.net.dataset]["mean_pixel"]
                _, loss, accuracy = sess.run([self.net.train_op, self.net.loss, self.net.accuracy],
                                             feed_dict={self.net.x: image_b, self.net.y: label_b})
                # label_b = np.squeeze(label_b, axis=0)
                # index = label_b.ravel()
                # image_size = [image_b.shape[1], image_b.shape[2], image_b.shape[3]]
                # color_image = Config[self.net.dataset]['palette'][index].reshape(image_size)
                # Image.fromarray(np.squeeze(color_image)).convert("RGB").save('1.png')
                # Image.fromarray(np.squeeze(label_b)).convert("L").save('2.bmp')
                # Image.fromarray(np.squeeze(np.asarray(image_b, dtype=np.uint8))).convert("RGB").save('3.png')
                if step % loss_freq == 0:
                    Tools.print_info("step {} loss : {}, acc : {}".format(step, loss, accuracy))
                if step % test_freq == 0:
                    self._test(sess, step)
                if step % model_freq == 0 or (step + 1) == epochs:
                    self.supervisor.saver.save(sess, os.path.join(self.model_path, "model_epoch_{}".format(step + 1)))
                pass
            pass
        pass

    def test(self, info):
        with self.supervisor.managed_session(config=self.config) as sess:
            self._test(sess=sess, info=info)
        pass

    def _test(self, sess, info):
        # load batch
        images, labels = self.data.get_test_data(0)
        prediction, is_correct, accuracy = sess.run([self.net.prediction, self.net.is_correct, self.net.accuracy],
                                                    feed_dict={self.net.x: images, self.net.y: labels})
        labels = np.squeeze(labels, axis=3) * (255 / self.data.class_number)
        prediction = np.squeeze(prediction, axis=3) * (255 / self.data.class_number)
        is_correct = np.squeeze(is_correct, axis=3) * 255
        Tools.print_info("acc is {}".format(accuracy))
        for index in range(len(prediction)):
            name = "{}_{}".format(info, index)
            Image.fromarray(images[index]).convert("RGB").save(os.path.join(self.result_path, "{}_i.png".format(name)))
            Image.fromarray(labels[index]).convert("L").save(os.path.join(self.result_path, "{}_l.png".format(name)))
            Image.fromarray(prediction[index]).convert("L").save(os.path.join(self.result_path, "{}_p.png".format(name)))
            Image.fromarray(is_correct[index]).convert("L").save(os.path.join(self.result_path, "{}_c.png".format(name)))
        pass

    pass


if __name__ == "__main__":

    cam_vid_data = CamVidData(class_number=11, batch_size=8, data_root_path="CamVid")
    seg_net = SegNet(cam_vid_data, learning_rate=1e-6, filter_size=16)
    runner = Runner(data=cam_vid_data, net=seg_net.seg_net(16), model_path=Tools.new_dir("model"), result_path=Tools.new_dir("result"))
    runner.train(epochs=100000, loss_freq=10, model_freq=100, test_freq=100)
    runner.test("test")

    pass