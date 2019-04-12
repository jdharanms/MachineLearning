#Imports
from configparser import ConfigParser
import numpy as np
import tensorflow as tf
from os import walk
from PIL import Image
import cv2
import time
from TensorflowModels import TfModels







class Predict_TF(object):
    """
    Predict will make a prediction based on an API request that has been sent, expecting JSON format.
      """

    def __init__(self):

        self.config = ConfigParser()
        self.config.read('./settings.ini')
        self.tf_models = TfModels()
        self.image_predict = self.config.get('File_System', 'Test_Images')
        self.standard_images = self.config.get('File_System', 'Standard_Images')
        self.machine_model = self.config.get('File_System', 'Model')


    def predict_digits(self):
        
        img_list_pred = []
        for (_, _, filenames) in walk(self.image_predict):
            for file in filenames:
                if (file.endswith(".png")):
                    image_path = self.image_predict+ file
                    img = Image.open(image_path)
                    img_list_pred.clear()
                    img_list_pred.append(img)
                    self.single_predict(img_list_pred,file)
        
    def single_predict(self,single_image,file):

        db_keys_digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        prediction = self.predict_image(single_image)

        for pred in prediction:
            predit = db_keys_digit[int(pred)]
            print("File %s : This image has been predicted as %s"%(file,predit))
            self.horiz_stack(predit,file)

    def horiz_stack(self,predicted,file_name):
 
        path_expected = self.image_predict+str(file_name)
        expected = cv2.imread(path_expected)
 
        expected = cv2.resize(expected, (0, 0), None, 10, 10)
 
        path_predicted = self.standard_images+str(predicted)+'.png'
        predicted = cv2.imread(path_predicted)
 
        predicted = cv2.resize(predicted, (0, 0), None, 10, 10)
 
        numpy_horizontal = np.hstack((expected, predicted))
 
        cv2.imshow('Expected vs Predicted', numpy_horizontal)
        cv2.waitKey()
        
    def predict_image(self, img_list):
        """
        This function is used to make a prediction and return an int or a list of ints that provide the prediction value.

        :param img_list: The list of images being passed through for evaluation.
        """

        my_graph = tf.Graph()

        # Convert the image into an numpy array, expected input into the solution.
        np_image_list = []

        for image_item in (img_list):
            image = np.asarray(image_item)
            image= image.reshape(28,28,1)
            np_image_list.append(image)

        np_list = np.array(np_image_list)
        with my_graph.as_default():
            num_classes = 10
            y_conv, x, _, keep_prob = self.tf_models.cnn_model(num_classes)

            saver = tf.train.Saver()
            model_file_path = self.machine_model+'digit_model.ckpt'

            with tf.Session(graph=my_graph) as sess:
                sess.run(tf.initialize_all_variables())
                saver.restore(sess, model_file_path)
                prediction = tf.argmax(y_conv, 1)
                return prediction.eval(feed_dict={x: np_list,keep_prob: 1.0}, session=sess)

if __name__ == '__main__':
    predict = Predict_TF()
    predict.predict_digits()       

        