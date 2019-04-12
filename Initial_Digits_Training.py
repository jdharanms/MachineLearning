"""Convolutional Neural Network Estimator for MNIST, built with tf.layers.

Adapted from:
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/layers/cnn_mnist.py

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from configparser import ConfigParser
import numpy as np
import tensorflow as tf
import shutil
import os


class Training(object):
    """
    Inital training on digits to get machine learning model to predict digits
    """
    
    def __init__(self):
        self.config = ConfigParser()
        self.config.read('./settings.ini')
        self.folder_name = self.config.get('File_System', 'Folder_Path')
    
    
    def digit_train(self):
        # Where to save Checkpoint(In the /output folder)
        resumepath = self.folder_name+"model/"
        filepath = self.folder_name+"Output_Club/"
        
        # Hyper-parameters
        # default batch size below
        # batch_size = 128
        batch_size = 64
        num_classes = 10
        # default no. of epochs below
        # num_epochs = 12
        num_epochs = 16
        learning_rate = 1e-4
        
        # If exists an checkpoint model, move it into the /output folder
        if os.path.exists(resumepath):
            shutil.copytree(resumepath, filepath)
        
        # Load training and eval data
        mnist = read_data_sets(train_dir='/input/MNIST_data', validation_size=0)
        train_data = mnist.train.images  # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images  # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
        
        
        train_data = train_data.reshape(train_data.shape[0],28,28,1)
        train_labels = train_labels.reshape(train_labels.shape[0])
        eval_data = eval_data.reshape(eval_data.shape[0],28,28,1)
        eval_labels = eval_labels.reshape(eval_labels.shape[0])
    
    
        def cnn_model_fn(features, labels, mode):
            """Model function for CNN."""
            # Input Layer
            # Reshape X to 4-D tensor: [batch_size, width, height, channels]
            # MNIST images are 28x28 pixels, and have one color channel
            input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
            input_layer = tf.cast(input_layer, tf.float32)
        
            # First convolutional layer
            w = tf.get_variable("w", shape=[3, 3, 1, 32], initializer=tf.contrib.layers.variance_scaling_initializer())
            b1 = tf.get_variable("b1", shape=[32], initializer=tf.constant_initializer(0.1))
            h_conv1 = tf.nn.relu(tf.nn.conv2d(input_layer, w, strides=[1,1,1,1],padding='SAME') + b1)
        
        
            # Second layer
            w2 = tf.get_variable("w2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.variance_scaling_initializer())
            b2 = tf.get_variable("b2", shape=[64], initializer=tf.constant_initializer(0.1))
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, w2, strides=[1,1,1,1],padding='SAME') + b2)
        
        
            # Pooling layer
            pool = tf.layers.max_pooling2d(inputs=h_conv2, pool_size=[2, 2], strides=2)
        
        
            # First dropout layer
            dropout = tf.layers.dropout(inputs=pool, rate=0.25)
        
        
            # Flattening tensor
            flat = tf.reshape(dropout, [-1, 14 * 14 * 64])
        
        
            # Dense Layer # 1
            # Densely connected layer with 128 neurons
            # Input Tensor Shape: [batch_size, 12 * 12 * 64] (batch_size, 9216)
            # Output Tensor Shape: [batch_size, 128]
            dense1 = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        
        
            # Second dropout layer
            # Add dropout operation; 0.5 probability that element will be kept
            dropout2 = tf.layers.dropout(
                inputs=dense1, rate=0.5)
        
        
            # Logits layer
            # Input Tensor Shape: [batch_size, 128]
            # Output Tensor Shape: [batch_size, 10]
            logits = tf.layers.dense(inputs=dropout2, units=num_classes)
        
            predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "classes": tf.argmax(input=logits, axis=1),
                # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                # `logging_hook`.
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
            # Inference (for TEST mode)
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
            # Calculate Loss (for both TRAIN and EVAL modes)
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes)
            # Cross Entropy
            loss = tf.losses.softmax_cross_entropy(
              onehot_labels=onehot_labels, logits=logits)
        
            # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                # AdamOptimizer
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
            # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = {
              "accuracy": tf.metrics.accuracy(
                  labels=labels, predictions=predictions["classes"])}
            return tf.estimator.EstimatorSpec(
              mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
        # Checkpoint Strategy configuration
        run_config = tf.contrib.learn.RunConfig(
            model_dir=filepath,
            keep_checkpoint_max=1)
        
        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
              model_fn=cnn_model_fn, config=run_config)
        
        # Keep track of the best accuracy
        best_acc = 0
        
        # Training for num_epochs
        for i in range(num_epochs):
            print("Begin Training - Epoch {}/{}".format(i+1, num_epochs))
            # Train the model for 1 epoch
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": train_data},
                y=train_labels,
                batch_size=batch_size,
                num_epochs=1,
                shuffle=True)
        
            mnist_classifier.train(
                input_fn=train_input_fn)
        
            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data},
                y=eval_labels,
                num_epochs=1,
                shuffle=False)
        
            eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        
            accuracy = eval_results["accuracy"] * 100
            # Set the best acc if we have a new best or if it is the first step
            if accuracy > best_acc or i == 0:
                best_acc = accuracy
                print ("=> New Best Accuracy {}".format(accuracy))
                if accuracy>98:
                    break
            else:
                print("=> Validation Accuracy did not improve")
            
            
        
if __name__ == '__main__':
    training = Training()
    training.digit_train()
