import tensorflow as tf
from configparser import ConfigParser

class TfModels(object):
    
    """
    A class that represents tensorflow models. 
    """

    def __init__(self):
        self.config = ConfigParser()
    
    def cnn_model(self,num_classes):
        """
        Convolutional Neural Network (CNN) classification model built using TensorFlow layers
        for recognising handwritten digits (N = 10). These models were pre-trained
        on benchmark data.
        """
    #Inputs will be grayscale images of size 28x28.
        x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x")
        #Actual labels
        y_ = tf.placeholder(tf.float32, [None, num_classes], name="y_")

    ## Defining the first convolutional layer

    # Defining weights (w), biases (b1) of the first convolutional layer with ReLU activation function (h_conv1). 
    # The variance scaling or He initialisation has been used to initialise weights as it is appropriate
    # for ReLU activation functions. Biases (b1) are initialised to a small random number (b1 = 0.1) to avoid
    # 'dead' neurons, i.e., neurons not firing during information processing/learning. A kernel of size 3x3 
    # has been used in this first convolutional layer.
        w = tf.get_variable("w", shape=[3, 3, 1, 32], initializer=tf.contrib.layers.variance_scaling_initializer())
        b1 = tf.get_variable("b1", shape=[32], initializer=tf.constant_initializer(0.1))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x, w, strides=[1,1,1,1],padding='SAME') + b1)
    
    ## Defining the second convolutional layer
    
    # Defining weights (w2), biases (b2) of the first convolutional layer with ReLU activation function (h_conv1). 
    # The variance scaling or He initialisation has been used to initialise weights as it is appropriate
    # for ReLU activation functions. Biases (b2) are initialised to a small random number (b1 = 0.1) to avoid
    # 'dead' neurons, i.e., neurons not firing during information processing/learning. A kernel of size 3x3 
    # has been used in this first convolutional layer.
        w2 = tf.get_variable("w2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.variance_scaling_initializer())
        b2 = tf.get_variable("b2", shape=[64], initializer=tf.constant_initializer(0.1))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, w2, strides=[1,1,1,1],padding='SAME') + b2)

    ## Defining a pooling layer 
    
    # Defining a pooling layer of 2x2 size for sub-sampling image data for subsequent classification.
    # The inputs of this layer are the outputs derived from the second convolutional layer.
        pool = tf.layers.max_pooling2d(inputs=h_conv2, pool_size=[2, 2], strides=2)
     
        ## Defining the first dropout layer

    # Defining a dropout layer for regularising the network, i.e., to help prevent
    # overfitting by dropping out some samples according to the dropout rate being defined,
    # thus avoiding training the network on noise for improving generalisation.
        dropout = tf.layers.dropout(inputs=pool, rate=0.25)
    
        # Flattening tensor
        flat = tf.reshape(dropout, [-1, 14 * 14 * 64])
    
        ## First dense layer

        # Densely connected layer with 128 neurons
        # Shape of the input tensor: [batch_size, 14 * 14 * 16 for letters or 14 * 14 * 64 for digits]
        # (batch_size, 3136) for letters or (batch_size, 12,544) for digits
        # Shape of the output tensor: [batch_size, 128]
        dense1 = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        ## Defining the second dropout layer

        # Adding dropout layer; dropout rate indicates the probability that the data will be kept
        dropout2 = tf.layers.dropout(
            inputs=dense1, rate=0.5)
    
        ## Defining a logits layer

        # Input Tensor Shape: [batch_size, 128]
        # Output Tensor Shape: [batch_size, 10 for digits, 21 for letters, as selected
	    # letters appearing in NINO are considered]
        logits = tf.layers.dense(inputs=dropout2, units=num_classes)
    
        keep_prob = tf.placeholder(tf.float32)

        y_conv = tf.nn.softmax(logits, name="softmax_tensor")

        return y_conv, x, y_, keep_prob