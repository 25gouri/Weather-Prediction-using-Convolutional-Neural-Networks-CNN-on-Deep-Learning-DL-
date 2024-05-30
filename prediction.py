import pickle
import os
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.preprocessing import image as image_utils
from keras.preprocessing import image
from keras import backend as K
import cv2

def img_prediction(test_image):
    K.clear_session()
    #data = []
    img_path = test_image
    #img_path = "cloudy1.jpg"
    testing_img = cv2.imread(img_path)
    # testing_img = cv2.imread(img_path)
    cv2.imwrite("..\\satelliteimage\static\\image_detection.jpg", testing_img)
    model = load_model('vgg16_model.h5')
    test_image = image.load_img(img_path, target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    #print("image2",test_image)
    test_image = np.expand_dims(test_image, axis=0)
    #print("image3",test_image)
    test_image /= 225
    prediction = model.predict(test_image)
    # print("prediction:",prediction)
    lb = pickle.load(open('label_transform.pkl_vgg16', 'rb'))
    # print("result:",lb.inverse_transform(prediction)[0])
    prediction = lb.inverse_transform(prediction)[0]
    print("pred:", prediction)
    K.clear_session()

    return prediction

#img_prediction()


