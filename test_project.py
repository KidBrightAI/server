#!/usr/bin/env python3
import sys, time, os, io, json
import numpy as np
import cv2

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# speech detection
from python_speech_features import mfcc

import tflite_runtime.interpreter as tflite

def preprocess(img):
  img = img.astype(np.float32)
  img = img / 255.
  img = img - 0.5
  img = img * 2.
  img = img[:, :, ::-1]
  return img

def classify(file,input_width, input_height):
  #im = cv2.imread(file)
  #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  #image_np = im.copy()
  im = Image.open(file)
  image_np = np.array(im)
  image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
  input_np = cv2.resize(image_np.copy(), (input_width, input_height))
  #cv2.imwrite("xxxx.jpg", input_np)
  input_np = preprocess(input_np)
  input_np = np.expand_dims(input_np, 0)
  interpreter.set_tensor(input_details["index"], input_np)
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details['index'])
  results = np.squeeze(output_data)
  out = results.argsort()[-1:][::-1]
  return out, results


if __name__ == '__main__':
    path = sys.argv[1]
    with open(path + '/project.json') as pjson:
      project = json.load(pjson)
    labels = project["project"]["project"]["modelLabel"]
    interpreter = tflite.Interpreter(path + '/model_edgetpu.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    _, input_height, input_width, _ = input_details['shape']
    output_details = interpreter.get_output_details()[0]
    corrected = 0
    failed = 0
    for dataset in project["dataset"]["dataset"]["data"]:
      dataset_id = dataset["id"]
      ext = dataset["ext"]
      classname = dataset["class"]
      filename = os.path.join(path,"raw_dataset",dataset_id + "_mfcc." + ext)
      out, results = classify(filename, input_width, input_width)
      label = labels[out[0]]
      print(dataset_id + " # " +classname + " --> " + label)

