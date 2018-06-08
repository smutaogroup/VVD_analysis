#!/bin/env python
import numpy as np
from sklearn.neural_network import MLPClassifier
from msmbuilder.io import load_meta, load_trajs
import sys
import tensorflow as tf

fold = int(sys.argv[1]) 
#alpha = 10 ** int(sys.argv[2])

crossvalid = np.load('crossvalid.npy').item()[fold]
train_index = crossvalid[0]
test_index  = crossvalid[1]

x = tf.placeholder(tf.float32, shape=[None, 148*148])
y_ = tf.placeholder(tf.float32, shape=[None, 8])

W = tf.Variable(tf.zeros([148*148,8]))
b = tf.Variable(tf.zeros([8]))

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for j in range(100):
		train_accuracy = 0
		for i in train_index:
			x_batch = np.load('alpha_carbon_sym/'+str(i)+'.npy')
			y_batch = np.load('labels_onehot/'+str(i)+'.npy')
			inter = int(len(x_batch)/100)
			for m in range(100):
				x_mini_batch = x_batch[m*inter:(m+1)*inter]
				y_mini_batch = y_batch[m*inter:(m+1)*inter]
				train_step.run(feed_dict={x: x_mini_batch, y_: y_mini_batch})
			train_accuracy += accuracy.eval(feed_dict={
				x:x_batch, y_:y_batch})
		train_accuracy = train_accuracy/ len(train_index) 
		print('step %d, training accuracy %g' % (j, train_accuracy))
	for i in test_index:
		x_test = np.load('alpha_carbon_sym/'+str(i)+'.npy')
		y_test = np.load('labels_onehot/'+str(i)+'.npy')
		print('test accuracy %g' % accuracy.eval(feed_dict={
			x: x_test, y_: y_test}))

