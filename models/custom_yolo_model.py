# -*- coding: utf-8 -*-
# This module is responsible for communicating with the outside of the yolo package.
# Outside the package, someone can use yolo detector accessing with this module.

import os
import time
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.fit import train
from utils.yolo.decoder import YoloDecoder
from utils.yolo.loss import YoloLoss

from .yolo_network import create_yolo_network
from utils.yolo.batch_gen import create_batch_generator
from utils.yolo.annotation import get_train_annotations_from_dataset, get_unique_labels
from utils.yolo.box import to_minmax
from utils.yolo.anchor import gen_anchor

def get_object_labels(ann_directory):
    files = os.listdir(ann_directory)
    files = [os.path.join(ann_directory, fname) for fname in files]
    return get_unique_labels(files)

def get_dataset_labels(dataset):
    labels = []
    for item in dataset:
        for bbox in item["annotate"]:
            if bbox["label"] and bbox["label"] not in labels:
                labels.append(bbox["label"])
    return labels

def get_json_layer_name(line):
    if line.startswith("{") and line.endswith("}"):
        layer = json.loads(line)
        if "name" in layer:
            return layer["name"]
    return None

def get_json_config(line):
    if line.startswith("{") and line.endswith("}"):
        layer = json.loads(line)
        return layer
    return None

def create_yolo(cmds,
                anchor_samples,
                labels,
                coord_scale = 1.0,
                object_scale = 1.0,
                class_scale = 5.0,
                no_object_scale = 1.0
                ):

    #1. check first layer must be input.
    if not cmds[0] or get_json_layer_name(cmds[0]) != "input":
        raise "First layer not input"
    #2. check last layer must be output.
    last = len(cmds) - 1
    if not cmds[last] or get_json_layer_name(cmds[last]) != "output":
        raise "Last layer not output"
    #3. check middle project must be yolo
    if not cmds[1] or get_json_layer_name(cmds[1]) != "yolo":
        raise "model is not yolo"
    #4. parse layer
    input_conf = json.loads(cmds[0])
    input_size = [input_conf["input_width"], input_conf["input_height"]]
    model_size = [input_conf["input_height"], input_conf["input_width"]]
    yolo_conf = json.loads(cmds[1])
    #5. parse output
    output_conf = json.loads(cmds[-1])
    
    # get anchors 
    anchors,iou_coverage = gen_anchor(anchor_samples,5,labels,input_size)
    print(anchors)
    
    n_classes = len(labels)
    n_boxes = int(len(anchors)/2)

    init_weight = "imagenet" if yolo_conf["weights"] == "imagenet" else None
    obj_thresh = yolo_conf["obj_thresh"]
    iou_thresh = yolo_conf["iou_thresh"]
    yolo_network = create_yolo_network(yolo_conf["arch"], model_size, n_classes, n_boxes, init_weight)

    yolo_loss = YoloLoss(yolo_network.get_grid_size(),
                         n_classes,
                         anchors,
                         coord_scale,
                         class_scale,
                         object_scale,
                         no_object_scale)

    yolo_decoder = YoloDecoder(anchors)
    yolo = YOLO(yolo_network, yolo_loss, yolo_decoder, labels, model_size)

    return yolo, input_conf, output_conf, anchors


class YOLO(object):
    def __init__(self,
                 yolo_network,
                 yolo_loss,
                 yolo_decoder,
                 labels,
                 input_size):

        self._yolo_network = yolo_network
        self._yolo_loss = yolo_loss
        self._yolo_decoder = yolo_decoder
        self._labels = labels
        self._input_size = input_size
        self.input_size = input_size
        self._norm = yolo_network._norm
        self.norm = yolo_network._norm

    def load_weights(self, weight_path, by_name=True):
        if os.path.exists(weight_path):
            print("Loading pre-trained weights for the whole model: ", weight_path)
            self._yolo_network.load_weights(weight_path, by_name=True)
        else:
            print("Failed to load pre-trained weights for the whole model. It might be because you didn't specify any or the weight file cannot be found")

    def predict(self, image, height, width, threshold=0.3):
        """
        # Args
            image : 3d-array (RGB ordered)
        
        # Returns
            boxes : array, shape of (N, 4)
            probs : array, shape of (N, nb_classes)
        """
        def _to_original_scale(boxes):
            #height, width = image.shape[:2]
            minmax_boxes = to_minmax(boxes)
            minmax_boxes[:,0] *= width
            minmax_boxes[:,2] *= width
            minmax_boxes[:,1] *= height
            minmax_boxes[:,3] *= height
            return minmax_boxes.astype(np.int)

        start_time = time.time()
        netout = self._yolo_network.forward(image)
        elapsed_ms = (time.time() - start_time) * 1000
        boxes, probs = self._yolo_decoder.run(netout, threshold)
        if len(boxes) > 0:
            boxes = _to_original_scale(boxes)
            #print(boxes, probs)
            return elapsed_ms, boxes, probs
        else:
            return elapsed_ms, [], []


    def train(self,
              train_dataset,
              train_img_folder,
              valid_dataset,
              valid_img_folder,
              nb_epoch,
              project_folder,
              batch_size,
              jitter,
              learning_rate, 
              train_times,
              valid_times,
              metrics,
              first_trainable_layer = None,
              callback_q = None,
              callback_sleep = None):

        # 1. get annotations        
        train_annotations, valid_annotations = get_train_annotations_from_dataset(self._labels,
                                        train_dataset,
                                        train_img_folder, 
                                        valid_dataset,
                                        valid_img_folder)
        # 1. get batch generator
        valid_batch_size = len(valid_annotations)*valid_times
        if valid_batch_size < batch_size: 
            raise ValueError("Not enough validation images: batch size {} is larger than {} validation images. Add more validation images or decrease batch size!".format(batch_size, valid_batch_size))
        
        train_batch_generator = self._get_batch_generator(train_annotations, batch_size, train_times, jitter=jitter)
        valid_batch_generator = self._get_batch_generator(valid_annotations, batch_size, valid_times, jitter=False)
        
        # 2. To train model get keras model instance & loss function
        model = self._yolo_network.get_model(first_trainable_layer)
        loss = self._get_loss_func(batch_size)
        
        # 3. Run training loop
        return train(model,
                loss,
                train_batch_generator,
                valid_batch_generator,
                learning_rate = learning_rate, 
                nb_epoch  = nb_epoch,
                project_folder = project_folder,
                first_trainable_layer = first_trainable_layer,
                network = self,
                metrics = metrics,
                report_callback = callback_q,
                callback_sleep = callback_sleep)

    def _get_loss_func(self, batch_size):
        return self._yolo_loss.custom_loss(batch_size)

    def _get_batch_generator(self, annotations, batch_size, repeat_times=1, jitter=True):
        """
        # Args
            annotations : Annotations instance
            batch_size : int
            jitter : bool
        
        # Returns
            batch_generator : BatchGenerator instance
        """
        batch_generator = create_batch_generator(annotations,
                                                 self._input_size,
                                                 self._yolo_network.get_grid_size(),
                                                 batch_size,
                                                 self._yolo_loss.anchors,
                                                 repeat_times,
                                                 jitter=jitter,
                                                 norm=self._yolo_network.get_normalize_func())
        return batch_generator
    
    