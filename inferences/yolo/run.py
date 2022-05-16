# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import os, sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from decoder import YoloDecoder
from box import to_minmax
#=========== config ===========

#yolo_model_tflite = "models/YOLO_best_mAP.tflite"
yolo_model = "YOLO_best_mAP_mobilenet_75_mAP_24.49.h5"
anchor = [0.24,0.29, 0.47,0.57, 0.80,0.96, 1.44,1.58, 3.14,2.54]
threshold = 0.5
input_size = (320,224)
labels = ["with_mask","without_mask"]

def prepare_image(img,input_size):
    input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    input_image = cv2.resize(input_image, input_size)
    input_image = input_image / 255
    input_image = input_image - 0.5
    input_image = input_image * 2.
    input_image = np.expand_dims(input_image, 0)
    return input_image

def bbox_to_xy(boxes,w,h):
    #height, width = image.shape[:2]
    minmax_boxes = to_minmax(boxes)
    minmax_boxes[:,0] *= w
    minmax_boxes[:,2] *= w
    minmax_boxes[:,1] *= h
    minmax_boxes[:,3] *= h
    return minmax_boxes.astype(np.int)

def draw_boxes(image, boxes, probs, labels):
    for box, classes in zip(boxes, probs):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 3)
        cv2.putText(image, 
                    '{}:  {:.2f}'.format(labels[np.argmax(classes)], classes.max()), 
                    (x1, y1 - 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image.shape[0], 
                    (0,0,255), 1)
    return image        

def run():
    video_reader = cv2.VideoCapture(0)
    decoder = YoloDecoder(anchor)
    model = load_model(yolo_model)

    # interpreter = tf.lite.Interpreter(model_path=yolo_model)
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # interpreter.allocate_tensors()
    # print(input_details)
    # print(output_details)

    while True:
        ret_val, image = video_reader.read()
        if ret_val == True:
            try:
                # STEP 1 -------- license plate detection --------#
                input = prepare_image(image,input_size)
                netout = model.predict(input)[0]
                boxes, probs = decoder.run(netout, threshold)
                if len(boxes) > 0:
                    boxes = bbox_to_xy(boxes,image.shape[1],image.shape[0])
                    box = boxes[0]
                    draw_boxes(image, boxes, probs, labels)
                cv2.imshow('video with bboxes', image)
            except:
                pass
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

if __name__ == '__main__':
    run()