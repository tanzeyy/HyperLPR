# coding : utf-8

import time
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
fontC = ImageFont.truetype("./sources/fonts/platech.ttf", 14, 0)

def drawRectBox(image,rect,addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2,cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex


import lpr as pr
import cv2
import numpy as np
import os

model = pr.LPR("./sources/model/cascade.xml","./sources/model/model12.h5","./sources/model/ocr_plate_all_gru.h5")

def detect(grr):
    # grr = cv2.imread(image)
    t1 = time.time()
    results = model.simple_detection_recognition(grr)
    t2 = time.time()
    print("Time used:", t2 - t1, "s")

    for pstr, confidence, rect in results:
        if confidence > 0.7:
            grr = drawRectBox(grr, rect, pstr + " " + str(round(confidence,3)))
            print ("plate_str:")
            print (pstr)
            print ("plate_confidence")
            print (confidence)

    return grr

if __name__ == "__main__":
    video_path = "/home/nicholas/data/Videos/Video1.mp4"
    cap = cv2.VideoCapture(video_path)
    while(1):
        _, frame = cap.read()
        height = frame.shape[0]
        width = frame.shape[1]
        scale = 1.0
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imshow("test", detect(frame))
        k = cv2.waitKey(2)
        if k & 0xFF == ord(' '):
            cv2.waitKey(0)
        elif k & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


