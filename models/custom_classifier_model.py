import time
import os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix #, ConfusionMatrixDisplay

from .feature import create_feature_extractor
from utils.classifier.batch_gen import create_datagen
from utils.fit import train
import tensorflow.keras.layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.mobilenet import preprocess_input


def mobilenet_normalize(self, image):
    image = image / 255.
    image = image - 0.5
    image = image * 2.

    return image	

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

def create_classifier(cmds):
    #1. check first layer must be input.
    if not cmds[0] or get_json_layer_name(cmds[0]) != "input":
        raise "First layer not input"
    #2. check last layer must be output.
    last = len(cmds) - 1
    if not cmds[last] or get_json_layer_name(cmds[last]) != "output":
        raise "Last layer not output"
    #3. check model contain mobilenet and mobilenet must be sit next to input
    found_mobilenet = False
    mobilenet_layer_index = -1
    for idx, cmd in enumerate(cmds):
        if get_json_layer_name(cmd) == "mobilenet":
            mobilenet_layer_index = idx
            found_mobilenet = True
            break 
    if found_mobilenet and mobilenet_layer_index != 1:
        raise "MobileNet must sit next to input layer"
    #4. parse layer
    input_node = None
    normalize = mobilenet_normalize
    x = None
    input_conf = json.loads(cmds[0])
    input_size = (input_conf["input_height"], input_conf["input_width"])
    if found_mobilenet:
        print("Found mobilenet layer")
        mobilenet_conf = json.loads(cmds[1])
        init_weight = "imagenet" if mobilenet_conf["weights"] == "imagenet" else None
        base_model = create_feature_extractor(mobilenet_conf["arch"], input_size, weights=init_weight)
        if not mobilenet_conf["trainable"]:
            print("mobilenet trainable layer is false")
            for layer in base_model.feature_extractor.layers:
                layer.trainable = False
        input_node = base_model.feature_extractor.inputs[0]
        x = base_model.feature_extractor.outputs[0]
        normalize = base_model.normalize
        cmds = cmds[2:]
    else:
        input_node = Input(shape=(input_conf["input_height"], input_conf["input_width"], 3))
        x = input_node
        cmds = cmds[1:]
        
    for cmd in cmds:
        if not cmd.startswith("{") or not cmd.endswith("}"): #json string
            x = eval(cmd + "(x)")
    
    #5. parse output
    output_conf = json.loads(cmds[-1])
    
    #6. build classifier
    model = Model(inputs = input_node, outputs = x, name = 'classifier')
    network = Classifier(model, input_size, normalize)
    return network, input_conf, output_conf

class Classifier(object):
    def __init__(self,
                 network,
                 input_size,
                 norm):
        self.network = network
        self.labels = []
        self.input_size = input_size
        self.norm = norm

    def load_weights(self, weight_path, by_name=False):
        if os.path.exists(weight_path):
            print("Loading pre-trained weights for the whole model: ", weight_path)
            self.network.load_weights(weight_path)
        else:
            print("Failed to load pre-trained weights for the whole model. It might be because you didn't specify any or the weight file cannot be found")


    def predict(self, img):

        start_time = time.time()
        Y_pred = np.squeeze(self.network(img, training = False))
        elapsed_ms = (time.time() - start_time)  * 1000

        y_pred = np.argmax(Y_pred)
        prob = Y_pred[y_pred]

        prediction = self.labels[y_pred]
        #print(Y_pred)
        return elapsed_ms, prob, prediction

    def train(self,
              img_folder,
              nb_epoch,
              project_folder,
              batch_size = 8,
              augumentation = False,
              learning_rate = 1e-4, 
              train_times = 1,
              valid_times = 1,
              valid_img_folder = "",
              first_trainable_layer = None,
              metrics = "val_loss",
              callback_q = None,
              callback_sleep = None):

        if metrics != "val_accuracy" and metrics != "val_loss":
            print("Unknown metric for Classifier, valid options are: val_loss or val_accuracy. Defaulting ot val_loss")
            metrics = "loss"
        #def create_datagen(train_folder, valid_folder, batch_size, input_size, project_folder, augumentation, norm):
        #train_generator = create_datagen(img_folder, batch_size, self.input_size, project_folder, augumentation, self.norm)
        #validation_generator = create_datagen(valid_img_folder, batch_size, self.input_size, project_folder, False, self.norm)
        train_generator, validation_generator, labels = create_datagen(img_folder, valid_img_folder, batch_size, self.input_size, project_folder, augumentation, self.norm)
        self.labels = labels
        model_layers, model_path = train(self.network,
                                        'categorical_crossentropy',
                                        train_generator,
                                        validation_generator,
                                        learning_rate, 
                                        nb_epoch, 
                                        project_folder,
                                        first_trainable_layer, 
                                        metrics = metrics,
                                        network = None,
                                        report_callback = callback_q,
                                        callback_sleep = callback_sleep)

        return model_layers, model_path
