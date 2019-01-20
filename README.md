# High Accuracy Chinese Plate Recognition Framework

### 介绍
Forked from HyperLPR by Zeusee. 

### 模型资源说明

+ cascade.xml  检测模型 - 目前效果最好的cascade检测模型
+ cascade_lbp.xml  召回率效果较好，但其错检太多
+ char_chi_sim.h5 Keras模型-可识别34类数字和大写英文字  使用14W样本训练 
+ char_rec.h5 Keras模型-可识别34类数字和大写英文字  使用7W样本训练 
+ ocr_plate_all_w_rnn_2.h5 基于CNN的序列模型
+ ocr_plate_all_gru.h5 基于GRU的序列模型从OCR模型修改，效果目前最好但速度较慢，需要20ms。
+ plate_type.h5 用于车牌颜色判断的模型
+ model12.h5 左右边界回归模型


### Python 依赖

+ Keras (>2.0.0)
+ Tensorflow(>1.1.x)
+ Numpy (>1.10)
+ Scipy (0.19.1)
+ OpenCV(>3.0)
+ Scikit-image (0.13.0)
+ PIL

### 模块说明

+ detection.py
用于检测车牌

+ recoginition.py
用于识别单张车牌

### 简单使用方式

```python
import lpr as pr
import cv2
import numpy as np
grr = cv2.imread("test_images/test_detection_1.jpg")
model = pr.LPR("model/cascade.xml","model/model12.h5","model/ocr_plate_all_gru.h5")

for pstr,confidence, rect in model.simple_detection_recognition(grr):
        if confidence > 0.7:
            image = drawRectBox(grr, rect, pstr + " " + str(round(confidence,3)))
            print("plate_str", pstr)
            print("plate_confidence", confidence)


cv2.imshow("image", image)
cv2.waitKey(0)

```

