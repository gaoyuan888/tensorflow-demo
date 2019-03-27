# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
tf.reset_default_graph()
class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim=10):
        # 网络参数
        self.input_dim = input_dim#输入维度
        self.seq_size = seq_size#时序长度
        self.hidden_dim = hidden_dim#隐层维度

        # 权重参数W与输入X及标签Y
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        # 均方误差求损失值，并使用梯度下降
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        # 建立saver保存模型
        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        #BasicLSTMCell基本的RNN类，建立hidden_dim个CELL
        cell = rnn.BasicLSTMCell(self.hidden_dim)#
        #dynamic_rnn动态RNN，cell生成好的cell类对象，self.x是一个张量，一般是三维张量[Batch_size,max_time(序列时间X0至Xt）),X具体输入]]
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        # tf.expand_dims,扩展维度，使得outputs与W相配备可以相乘
        #W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        print('W_out',self.W_out)
        tf_expand = tf.expand_dims(self.W_out, 0)
        print('tf_expand',tf_expand)
        tf_tile = tf.tile(tf_expand, [num_examples, 1, 1])
        print('tf_tile',tf_tile)
        print('outputs',outputs)
        out = tf.matmul(outputs, tf_tile) + self.b_out
        # tf.squeeze 压缩为1的维度
        print(out)
        out = tf.squeeze(out)
        print(out)
        return out

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()#变量可重复利用
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                _, mse = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if i % 100 == 0:
                    print(i, mse)
            save_path = self.saver.save(sess, './model')
            print('Model saved to {}'.format(save_path))

    def test(self, test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, './model')
            output = sess.run(self.model(), feed_dict={self.x: test_x})
            return output


if __name__ == '__main__':
    predictor = SeriesPredictor(input_dim=1, seq_size=4, hidden_dim=10)
    train_x = [[[1], [2], [5], [6]],
               [[5], [7], [7], [8]],
               [[3], [4], [5], [7]]]
    train_y = [[1, 3, 7, 11],
               [5, 12, 14, 15],
               [3, 7, 9, 12]]
    predictor.train(train_x, train_y)#传入x,y开始训练网络

    test_x = [[[1], [2], [3], [4]],  # 1, 3, 5, 7
              [[4], [5], [6], [7]]]  # 4, 9, 11, 13
    actual_y = [[[1], [3], [5], [7]],
                [[4], [9], [11], [13]]]
    pred_y = predictor.test(test_x)
    
    print("\n开始测试!\n")
    
    for i, x in enumerate(test_x):
        print("我们当前的输入 {}".format(x))
        print("应该得到的输出 {}".format(actual_y[i]))
        print("训练模型得到的输出 {}\n".format(pred_y[i]))