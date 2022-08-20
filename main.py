from flask import Flask, render_template, request, copy_current_request_context, jsonify, send_file, send_static_file
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS, cross_origin
import threading, queue
import ctypes
import zipfile

import requests
import atexit
from pathlib import Path

import sys, json, os, time, logging, random, shutil, tempfile, subprocess, re, platform, io
import numpy as np
import cv2
import utils.helper as helper
sys.path.append(".")

from models.custom_classifier_model import create_classifier
from models.custom_yolo_model import create_yolo, get_dataset_labels
from convert import Converter

import gdown

from tensorflow import keras 
from tensorflow.keras import backend as K 

#from keras import backend as K 

BACKEND = "EDGE" if platform.node() == "raspberrypi" else "COLAB"
SUDO_PASS = "raspberry"
print("BACKEND : " + BACKEND)
PROJECT_PATH = "./" if BACKEND == "COLAB" else "./projects"
PROJECT_FILENAME = "project.json"
TRAIN_FOLDER = "train"
TEST_FOLDER = "test"
VALIDATE_FOLDER = "valid"
DATASET_FOLDER = "dataset"
RAW_DATASET_FOLDER = "raw_dataset"
OUTPUT_FOLDER = "output"
TEMP_FOLDER = "temp"

STAGE = 0 #0 none, 1 = prepare dataset, 2 = training, 3 = trained, 4 = converting, 5 converted
report_queue = queue.Queue()
train_task = None
report_task = None

app = Flask(__name__)
#Set this argument to``'*'`` to allow all origins, or to ``[]`` to disable CORS handling.
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins = "*")
CORS(app)
#CRITICAL, ERROR, WARNING, INFO, DEBUG
logging.basicConfig(level=logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

@app.route('/wifi', methods=["GET","POST"])
def on_wifi():
    if request.method == 'GET':
        result = subprocess.check_output(f"echo '{SUDO_PASS}' | sudo -S wifi scan", shell=True)
        network = result.decode('ascii')
        match = re.findall(r"-[0-9]+\s+(.*?)\s",network)
        return jsonify({"result":"OK",  "data" : match })
    if request.method == 'POST':
        return jsonify({"result":"OK"})

@app.route('/wifi_current', methods=["GET"])
def on_current_wifi():
    if request.method == 'GET':
        result = subprocess.check_output(f"wpa_cli -i wlan0 status", shell=True)
        network = result.decode('ascii')
        print(network)
        results = dict(x.split("=") for x in network.split("\n") if len(x.split("=")) == 2)
        return jsonify({"result":"OK",  "data" : results })

@app.route("/list_project", methods=["GET"])
def handle_list_project():
    res = []
    for proj_id in os.listdir(PROJECT_PATH):
        if not proj_id.startswith("project-"):
            continue
        project_all = helper.read_json_file(os.path.join(PROJECT_PATH,proj_id,PROJECT_FILENAME))
        project = project_all["project"]["project"]
        info = {
            "name": project["name"], 
            "description": project["description"], 
            "id": project["id"],
            "projectType": project["projectType"],
            "projectTypeTitle": project["projectTypeTitle"],
            "lastUpdate": project["lastUpdate"]
        }
        res.append(info)
    return jsonify({"result":"OK", "projects" : res})

@app.route('/save_project', methods=["GET"])
def on_save():
    return jsonify({"result":"OK"})

@app.route("/load_project", methods=["POST"])
def handle_load_project():
    data = request.get_json()
    res = {}
    project_id = data["project_id"]
    project_file = os.path.join(PROJECT_PATH,project_id,PROJECT_FILENAME)
    if os.path.exists(project_file):
        project_info = helper.read_json_file(project_file)
        res = project_info
    return jsonify({"project_data" : res})

@app.route("/download_project", method=["POST"])
def handle_download_project():
    data = request.get_json()
    project_id = data["project_id"]
    project_zip_file = os.path.join(PROJECT_PATH,project_id+".zip")
    project_target_dir = os.path.join(PROJECT_PATH,project_id)
    shutil.make_archive(project_zip_file, 'zip', project_target_dir)
    return send_static_file(project_zip_file)

@app.route('/delete_project', methods=["POST"])
def on_delete_project():
    return jsonify({"result":"OK"})

@app.route('/run', methods=["POST"])
def on_run():
    return jsonify({"result":"OK"})

@app.route("/sync_project", methods=["POST"])
#@cross_origin(origin="*")
def sync_project():
    data = request.get_json() #content = request.get_json(silent=True)
    # ======== init project =======#
    project = data['project']['project']
    # check project dir exists
    project_path = os.path.join(PROJECT_PATH, project['id'])
    helper.create_not_exist(project_path)
    # write project file
    project_file = os.path.join(project_path,PROJECT_FILENAME)
    with open(project_file, 'w') as json_file:
        json.dump(data, json_file)
    # ========= sync project ======#
    # sync dataset
    RAW_PROJECT_DATASET = os.path.join(project_path,RAW_DATASET_FOLDER)
    helper.create_not_exist(RAW_PROJECT_DATASET) 
    dataset = data["dataset"]["dataset"]
    if project["projectType"] == "VOICE_CLASSIFICATION":
        needed_filename = [i["id"]+"_mfcc.jpg" for i in dataset["data"]]
    else:
        needed_filename = [i["id"]+"."+i["ext"] for i in dataset["data"]]
    needed_files = helper.sync_files(RAW_PROJECT_DATASET, needed_filename)
    res = "OK" if len(needed_files) == 0 else "SYNC"
    return jsonify({"result" : res, "needed" : needed_files})
    # =========================== #

@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST":
        files = request.files.getlist("dataset")
        project_id = request.form['project_id']
        dataset_raw_path = os.path.join(PROJECT_PATH, project_id, RAW_DATASET_FOLDER)
        for file in files:
            target_path = os.path.join(dataset_raw_path,file.filename)
            file.save(target_path)
    return jsonify({"result":"OK"})


def training_task(data, q):
    global STAGE, current_model
    K.clear_session()
    try:
        # 1 ========== prepare project ========= #
        STAGE = 1
        q.put({"time":time.time(), "event": "initial", "msg" : "Start training step 1 ... prepare dataset"})
        project_id = data["project_id"]
        project_file = os.path.join(PROJECT_PATH, project_id, PROJECT_FILENAME)
        project = helper.read_json_file(project_file)
        model = project["project"]["project"]["model"]
        cmd_code = model["code"]
        config = helper.parse_json(cmd_code)
        q.put({"time":time.time(), "event": "initial", "msg" : "target project id : "+project_id})
        # 2 ========== prepare dataset ========= #
        train_type = config["model"]
        raw_dataset_path = os.path.join(PROJECT_PATH, project_id, RAW_DATASET_FOLDER)
        dataset = project["dataset"]["dataset"]["data"]
        
        #remove corrupted file
        
        q.put({"time":time.time(), "event": "initial", "msg" : "check corrupted files ..."})
        print("step 0 check corrupted files")
        corrupted_file = helper.check_and_remove_corrupted_image(raw_dataset_path)
        q.put({"time":time.time(), "event": "initial", "msg" : "found corrupted image file : " + str(len(corrupted_file))})
        print("found corrupted image file : ")
        print(corrupted_file)
        if len(corrupted_file) > 0:
            dataset = [el for el in dataset if (el["id"]+"."+el["ext"]) not in corrupted_file]
        q.put({"time":time.time(), "event": "initial", "msg" : "corrupted file has been removed from dataset"})
        
        random.shuffle(dataset)
        #################################################################
        # WARNING !!! we didn't distributed train/test split each class #
        #################################################################
        q.put({"time":time.time(), "event": "initial", "msg" : "prepare dataset ... contain " + str(len(dataset)) + " files" })
        train, valid = np.split(dataset,[int(len(dataset) * (config["train_rate"]/100.0))])
        print("training length : " + str(len(train)))
        print("validate length : " + str(len(valid)))
        q.put({"time":time.time(), "event": "initial", "msg" : "train length : " + str(len(train)) })
        q.put({"time":time.time(), "event": "initial", "msg" : "validate length : " + str(len(valid))})
        train_dataset_path = os.path.join(PROJECT_PATH, project_id, DATASET_FOLDER, TRAIN_FOLDER)
        valid_dataset_path = os.path.join(PROJECT_PATH, project_id, DATASET_FOLDER, VALIDATE_FOLDER)
        shutil.rmtree(train_dataset_path, ignore_errors=True)
        shutil.rmtree(valid_dataset_path, ignore_errors=True)
        
        if train_type == "mobilenet":
            q.put({"time":time.time(), "event": "initial", "msg" : "training type : classification(mobilenet)"})
            print("step 1 prepare dataset")
            #create folder with label
            shutil.rmtree(train_dataset_path, ignore_errors=True)
            shutil.rmtree(valid_dataset_path, ignore_errors=True)
            if project["project"]["project"]["projectType"] == "VOICE_CLASSIFICATION": # training file end with id_mfcc.jpg
                labels = helper.move_dataset_file_to_folder(train, raw_dataset_path, train_dataset_path, "_mfcc", "jpg")
            else:
                labels = helper.move_dataset_file_to_folder(train, raw_dataset_path, train_dataset_path)
            print("train data moved to : " + train_dataset_path)
            q.put({"time":time.time(), "event": "initial", "msg" : "train data moved to : " + train_dataset_path})

            if project["project"]["project"]["projectType"] == "VOICE_CLASSIFICATION": # test file end with id_mfcc.jpg
                helper.move_dataset_file_to_folder(valid, raw_dataset_path, valid_dataset_path, "_mfcc", "jpg")
            else:
                helper.move_dataset_file_to_folder(valid, raw_dataset_path, valid_dataset_path)
            
            print("validate data moved to : " + valid_dataset_path)
            q.put({"time":time.time(), "event": "initial", "msg" : "validate data moved to : " + valid_dataset_path})
            print("labels : ")
            print(labels)
            q.put({"time":time.time(), "event": "initial", "msg" : "train label : " + ",".join(labels)})
            
            cmd_lines = cmd_code.split("\n")
            current_model, input_conf, output_conf = create_classifier(cmd_lines)
            output_folder_path = os.path.join(PROJECT_PATH, project_id, OUTPUT_FOLDER)
            shutil.rmtree(output_folder_path, ignore_errors=True)
            helper.create_not_exist(output_folder_path)
            
            #download pretrained model
            if input_conf["pretrained"] and input_conf["pretrained"].startswith("https://drive.google.com"):
                q.put({"time":time.time(), "event": "initial", "msg" : "download pretrained model : " + input_conf["pretrained"]})
                pretrained_model_file = os.path.join(output_folder_path,"pretrained_model.h5")
                gdown.download(input_conf["pretrained"],pretrained_model_file,quiet=False)
                current_model.load_weights(pretrained_model_file)

            # stringlist = []
            # model.network.summary(print_fn=lambda x: stringlist.append(x))
            # model_summary = "\n".join(stringlist)
            # q.put({"time":time.time(), "event": "initial", "msg" : "model network : \n" + model_summary})
            
            STAGE = 2
            current_model.train(
                train_dataset_path,
                input_conf["epochs"],
                output_folder_path,
                batch_size = input_conf["batch_size"],
                augumentation = True,
                learning_rate = input_conf["learning_rate"], 
                train_times = input_conf["train_times"],
                valid_times = input_conf["valid_times"],
                valid_img_folder = valid_dataset_path,
                first_trainable_layer = None,
                metrics = output_conf["save_on"],
                callback_q = q,
                callback_sleep = None)
            STAGE = 3
            print("finish traing")

        elif train_type == "yolo":
            q.put({"time":time.time(), "event": "initial", "msg" : "training type : Yolo object detection"})
            # get label
            labels = get_dataset_labels(train)
            print("labels : ")
            print(labels)
            q.put({"time":time.time(), "event": "initial", "msg" : "train label : " + ",".join(labels)})
            
            cmd_lines = cmd_code.split("\n")

            current_model, input_conf, output_conf, anchors = create_yolo(cmd_lines,dataset, labels)
            q.put({"time":time.time(), "event": "initial", "msg" : "model created "})
            q.put({"time":time.time(), "event": "initial", "msg" : "anchors = " + ", ".join(str(el) for el in anchors)})
            output_folder_path = os.path.join(PROJECT_PATH, project_id, OUTPUT_FOLDER)
            shutil.rmtree(output_folder_path, ignore_errors=True)
            helper.create_not_exist(output_folder_path)

            # write anchors to files
            helper.write_text_file(os.path.join(output_folder_path,"anchors.txt"), ",".join(str(el) for el in anchors))
            
            if input_conf["pretrained"] and input_conf["pretrained"].startswith("https://drive.google.com"):
                q.put({"time":time.time(), "event": "initial", "msg" : "download pretrained model : " + input_conf["pretrained"]})
                pretrained_model_file = os.path.join(output_folder_path,"pretrained_model.h5")
                gdown.download(input_conf["pretrained"],pretrained_model_file,quiet=False)
                current_model.load_weights(pretrained_model_file)
                
            STAGE = 2

            current_model.train(train,
                raw_dataset_path,
                valid,
                raw_dataset_path,
                input_conf["epochs"],
                output_folder_path,
                batch_size = input_conf["batch_size"],
                jitter = True, #augumentation
                learning_rate = input_conf["learning_rate"], 
                train_times = input_conf["train_times"],
                valid_times = input_conf["valid_times"],
                metrics = output_conf["save_on"],
                callback_q = q,
                callback_sleep = None)

            STAGE = 3
            q.put({"time":time.time(), "event": "train_end", "msg" : "Train ended", "matric" : None})            
            # print("finish traing")

    finally:
        print("Thread ended")

def kill_thread(target_thread):
    target_thread_id = None
    if hasattr(target_thread, '_thread_id'):
        target_thread_id = target_thread._thread_id
    else:
        for id, thread in threading._active.items():
            if thread is target_thread:
                target_thread_id = id
    if target_thread_id != None: 
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(target_thread_id,ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(target_thread_id, 0)
            print('Exception raise failure')
        else:
            print('Thread exited')

@app.route("/start_training", methods=["POST"])
def start_training():
    global train_task, report_queue
    print("start training process")
    data = request.get_json()
    train_task = threading.Thread(target=training_task, args=(data,report_queue,))
    train_task.start()
    return jsonify({"result" : "OK"})

@app.route("/terminate_training", methods=["POST"])
def terminate_training():
    global train_task, report_task, report_queue
    print("terminate current training process")
    if train_task and train_task.is_alive():
        kill_thread(train_task)
        print("send kill command")
    #if report_task and report_task.is_alive():
    #    report_queue.put({"time": time.time(), "event" : "terminate", "msg":"Training terminated"})
    time.sleep(3)
    return jsonify({"result" : "OK"})

@app.route("/convert_model", methods=["POST"])
def handle_convert_model():
    print("convert model")
    data = request.get_json()
    res = {}
    project_id = data["project_id"]
    if not project_id:
        return "Fail"
    output_path = os.path.join(PROJECT_PATH, project_id, "output")
    files = [os.path.join(output_path,f) for f in os.listdir(output_path) if f.endswith(".h5")]
    if len(files) <= 0:
        return "Fail"
    
    project_file = os.path.join(PROJECT_PATH, project_id, PROJECT_FILENAME)
    project = helper.read_json_file(project_file)
    model = project["project"]["project"]["model"]
    cmd_code = model["code"]
    config = helper.parse_json(cmd_code)
    raw_dataset_path = os.path.join(PROJECT_PATH, project_id, RAW_DATASET_FOLDER)
    output_model_path = os.path.join(PROJECT_PATH, project_id, OUTPUT_FOLDER)

    converter = Converter("edgetpu", config["arch"], raw_dataset_path)
    converter.convert_model(files[0])
    
    shutil.make_archive(os.path.join(PROJECT_PATH, project_id, "model"), 'zip', output_model_path)

    return jsonify({"result" : "OK"})

@app.route("/download_model", methods=["GET"])
def handle_download_model():
    print("download model file")
    project_id = request.args.get("project_id")
    model_export = os.path.join(PROJECT_PATH,project_id,"model.zip")
    return send_file(model_export, as_attachment=True)

@app.route("/inference_image", methods=["POST"])
def handle_inference_model():
    global STAGE, current_model
    if 'image' not in request.files:
        return "No image"
    if STAGE < 3:
        return "Training not success yet :" + str(STAGE)
    
    tmp_img = request.files['image']
    project_id = request.form['project_id']
    model_type = request.form['type']

    if not tmp_img:
        return "Image null or something"
    
    target_file_path = os.path.join(PROJECT_PATH, project_id, TEMP_FOLDER)
    helper.create_not_exist(target_file_path) 
    target_file = os.path.join(target_file_path, tmp_img.filename)
    tmp_img.save(target_file)    

    if model_type == "classification":
        orig_image, img = helper.prepare_image(target_file, current_model, current_model.input_size)
        elapsed_ms, prob, prediction = current_model.predict(img)
        return jsonify({"result" : "OK","prediction":prediction, "prob":np.float64(prob)})
    elif model_type == "detection":
        threshold = float(request.form['threshold'])
        orig_image, input_image = helper.prepare_image(target_file, current_model, current_model._input_size)
        height, width = orig_image.shape[:2]
        prediction_time, boxes, probs = current_model.predict(input_image, height, width, threshold)
        labels = current_model._labels
        bboxes = []
        for box, classes in zip(boxes, probs):
            x1, y1, x2, y2 = box
            bboxes.append({
                "x1" : np.float64(x1), 
                "y1" : np.float64(y1), 
                "x2" : np.float64(x2), 
                "y2" : np.float64(y2), 
                "prob" : np.float64(classes.max()), 
                "label" : labels[np.argmax(classes)]
            })
        return jsonify({"result" : "OK", "boxes": bboxes})
    else:
        return jsonify({"result" : "FAIL","reason":"model type not specify"})


@socketio.on('connect')
def client_connect():
    global report_task, report_queue
    print("client connected")
    # check prev thread are alive
    if report_task and report_task.is_alive():
        report_queue.put({"time": time.time(), "event" : "terminate", "msg":"terminate client"})
        time.sleep(1)
    @copy_current_request_context
    def train_callback(q):
        while True:
            try:
                report = q.get(block=True, timeout=None)
                if report["event"] == "train_end":
                    print("traning end")
                    #break
                if report["event"] == "terminate":
                    print("terminate thread called")
                    break
                else:
                    emit("training_progress",report)
            except Queue.Empty:
                continue
            except:
                print("Unknow Error")
                break 
    report_task = threading.Thread(target=train_callback, args=(report_queue,))
    report_task.start()

@socketio.on('disconnect')
def client_disconnect():
    print('Client disconnected')

def _run_ngrok(port,token):
    command = "ngrok"
    ngrok_path = str(Path(tempfile.gettempdir(), "ngrok"))
    _download_ngrok(ngrok_path)
    executable = str(Path(ngrok_path, command))
    os.chmod(executable, 0o777)
    subprocess.run([executable, 'authtoken', token])
    time.sleep(3)
    ngrok = subprocess.Popen([executable, 'http', str(port)])
    atexit.register(ngrok.terminate)
    localhost_url = "http://localhost:4040/api/tunnels"  # Url with tunnel details
    time.sleep(1)
    tunnel_url = requests.get(localhost_url).text  # Get the tunnel information
    j = json.loads(tunnel_url)

    tunnel_url = j['tunnels'][0]['public_url']  # Do the parsing of the get
    #tunnel_url = tunnel_url.replace("https", "http")
    if tunnel_url.startswith("http://"):
        tunnel_url = tunnel_url.replace("http://","https://")
    return tunnel_url


def _download_ngrok(ngrok_path):
    if Path(ngrok_path).exists():
        return
    url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip"
    download_path = _download_file(url)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(ngrok_path)

def _download_file(url):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    download_path = str(Path(tempfile.gettempdir(), local_filename))
    with open(download_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return download_path

def start_ngrok(port,token):
    ngrok_address = _run_ngrok(port, token)
    print(f" * Server Address : {ngrok_address}")

def run_ngrok(port,token):
    thread = threading.Timer(1, start_ngrok, args=(port,token,))
    thread.setDaemon(True)
    thread.start()

if __name__ == '__main__':
    len_arg = len(sys.argv)
    if len_arg > 2:
        if sys.argv[1] == "ngrok" and sys.argv[2]:
            print("=== start ngrok ===")
            run_ngrok(5000,sys.argv[2])
    app.run(host="0.0.0.0",debug=True)
    #socketio.run(app,debug=True,port=8888)
    #data = {"project_id" : "project-sss-mRshh0"}
    #training_task(data,report_queue)