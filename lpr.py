# coding=utf-8

import cv2
from detection import Detector
from recoginition import Recognizer

class LPR(Detector, Recognizer):

    def __init__(self, model_detection, model_finemapping, model_seq_rec):
        Detector.__init__(self, model_detection, model_finemapping)
        Recognizer.__init__(self, model_seq_rec)

    def simple_detection_recognition(self, image):
        rects = self.detect_plates(image)
        results = []
        for plate, rect in rects:
            result, confidence = self.recognize_one(plate)
            results.append([result, confidence, rect])
        return results


if __name__ == "__main__":
    lpr = LPR("./sources/model/cascade.xml", "./sources/model/model12.h5", "./sources/model/ocr_plate_all_gru.h5")
    img = cv2.imread("./test_images/test_detection_1.jpg")
    results = lpr.simple_detection_recognition(img)
    for res, conf, rect in results:
        print(res)
        print("Confidence:", conf)
