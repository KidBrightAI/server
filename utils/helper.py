import json, os, copy, time, logging, random, shutil
import numpy as np
import cv2

def prepare_image(img_path, network, input_size):
    orig_image = cv2.imread(img_path)
    input_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) 
    input_image = cv2.resize(input_image, (input_size[1], input_size[0]))
    input_image = network.norm(input_image)
    input_image = np.expand_dims(input_image, 0)
    return orig_image, input_image

def create_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def write_file(file,data):
    with open(file, 'wb') as f:
        f.write(data) 

def read_json_file(file):
    if os.path.exists(file):
        with open(file) as f:
            return json.load(f)

def sync_files(path,needed_files):
    all_files = os.listdir(path)
    removed = 0
    requested = []
    for item in all_files:
        target_file = os.path.join(path,item)
        if not os.path.isfile(target_file):
            continue
        # remove not needed
        if item not in needed_files:
            os.remove(target_file)
            removed += 1
    for item in needed_files:
        if item not in all_files:
            requested.append(item)
    return requested

def check_and_remove_corrupted_image(path):
    all_files = os.listdir(path)
    corrupted_file = []
    for item in all_files:
        target_file = os.path.join(path,item)
        img = cv2.imread(target_file)
        if (type(img) is not np.ndarray): 
            corrupted_file.append(item)
            os.remove(target_file)
    return corrupted_file

def parse_json(cmd):
    lines = cmd.split("\n")
    config = {}
    for line in lines:
        if line.startswith("{") and line.endswith("}") :
            config_json = json.loads(line)
            config.update(config_json)
    return config

def move_dataset_file_to_folder(dataset, dataset_path, target_path):
    dirs = []
    for item in dataset:
        class_name = item["class"]
        if class_name not in dirs:
            dirs.append(class_name)
            create_not_exist(os.path.join(target_path, class_name))

        filename = item["id"] + "." + item["ext"]
        src_file = os.path.join(dataset_path, filename)
        des_file = os.path.join(target_path, class_name, filename)
        shutil.copyfile(src_file, des_file)
    return dirs

