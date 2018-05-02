#coding=utf-8

import cv2
import numpy as np
from keras import backend as K
from keras.models import *
from keras.layers import *

class Detector:
    
    def __init__(self, model_detection, model_finemapping):
        self.watch_cascade = cv2.CascadeClassifier(model_detection)
        self.modelFineMapping = self.model_fine_mapping(model_finemapping)

    def model_fine_mapping(self, model_finemapping):
        input = Input(shape=[16, 66, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
        x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = Activation("relu", name='relu1')(x)
        x = MaxPool2D(pool_size=2)(x)
        x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = Activation("relu", name='relu2')(x)
        x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
        x = Activation("relu", name='relu3')(x)
        x = Flatten()(x)
        output = Dense(2,name = "dense")(x)
        output = Activation("relu", name='relu4')(output)
        model = Model([input], [output])
        model.load_weights(model_finemapping)
        return model

    def compute_safe_region(self, shape, bounding_rect):
        top = bounding_rect[1] # y
        bottom  = bounding_rect[1] + bounding_rect[3] # y + h
        left = bounding_rect[0] # x
        right = bounding_rect[0] + bounding_rect[2] # x + w
        min_top = 0
        max_bottom = shape[0]
        min_left = 0
        max_right = shape[1]
        if top < min_top:
            top = min_top
        if left < min_left:
            left = min_left
        if bottom > max_bottom:
            bottom = max_bottom
        if right > max_right:
            right = max_right
        return [left, top, right - left, bottom - top]

    def crop_image(self, image, rect):
        x, y, w, h = self.compute_safe_region(image.shape, rect)
        return image[y:y+h,x:x+w]

    def detect_plate_rough(self, image_gray, resize_h=720, en_scale=1.08 , top_bottom_padding_rate=0.05):
        if top_bottom_padding_rate > 0.2:
            print ("error:top_bottom_padding_rate > 0.2:",top_bottom_padding_rate)
            exit(1)
        height = image_gray.shape[0]
        padding = int(height*top_bottom_padding_rate)
        scale = image_gray.shape[1] / float(image_gray.shape[0])
        image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))
        image_color_cropped = image[padding:resize_h-padding,0:image_gray.shape[1]]
        image_gray = cv2.cvtColor(image_color_cropped,cv2.COLOR_RGB2GRAY)
        watches = self.watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),maxSize=(36*40, 9*40))
        cropped_images = []
        for (x, y, w, h) in watches:
            x -= w * 0.14
            w += w * 0.28
            y -= h * 0.15
            h += h * 0.3
            cropped = self.crop_image(image_color_cropped, (int(x), int(y), int(w), int(h)))
            cropped_images.append([cropped,[x, y+padding, w, h]])
        return cropped_images

    def finemapping_vertical(self, image, rect):
        resized = cv2.resize(image,(66,16))
        resized = resized.astype(np.float)/255
        res_raw= self.modelFineMapping.predict(np.array([resized]))[0]
        res  = res_raw * image.shape[1]
        res = res.astype(np.int)
        H,T = res
        H -= 3
        if H < 0:
            H = 0
        T += 2;
        if T >= image.shape[1]-1:
            T = image.shape[1]-1
        rect[2] -=  rect[2] * (1 - res_raw[1] + res_raw[0])
        rect[0] += res[0]
        image = image[:,H:T+2]
        image = cv2.resize(image, (136, 36))
        return image, rect

    def detect_plates(self, image):
        images = self.detect_plate_rough(image, image.shape[0], top_bottom_padding_rate=0.1)
        rects_set = []
        for index, plate in enumerate(images):
            plate, rect = plate
            image_rgb, rect_refine = self.finemapping_vertical(plate, rect)
            rects_set.append([image_rgb, rect_refine])
        return rects_set




if __name__ == "__main__":
    detector = Detector("./sources/model/cascade.xml", "./sources/model/model12.h5")
    image = cv2.imread("./test_images/test_detection_2.jpg")
    rects = detector.detect_plates(image)
    for image, rect in rects:
        cv2.imshow("image", image)
        cv2.waitKey(0)