cd
git clone https://github.com/KidBrightAI/server kbai-server
cd kbai-server
#sudo raspi-config ->extend partion
gdown --id 11mujzVaFqa7R1_lB7q0kVPW22Ol51MPg
pip3 install tensorflow-2.2.0-cp37-cp37m-linux_armv7l.whl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
pip3 install tensorflowjs

# pip3 install https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.4.0-rc2/tensorflow-2.4.0rc2-cp37-none-linux_armv7l.whl
# pip3 install https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl
# pip3 install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl
# pip3 install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.3.0/tensorflow-2.3.0-cp37-none-linux_armv7l.whl
# https://github.com/easonlai/raspberry-pi3-tensorflow2/blob/main/tensorflow-2.4.0-cp37-none-linux_armv7l.whl
# pip3 install -r requirements_rpi.txt
# tflite_convert --keras_model_file=projects/project-fsdfs-h6fMJm/output/Classifier_best_val_accuracy.h5 --output_file=./projects/project-fsdfs-h6fMJm/output/test.tflite
# #sudo docker run -it --rm -v /home/pi/kbai-server:/home armindocachada/tensorflow2-raspberrypi4:2.3.0-cp35-none-linux_armv7l tflite_convert --keras_model_file=/home/projects/project-fsdfs-h6fMJm/output/Classifier_best_val_accuracy.h5 --output_file=/home/projects/project-fsdfs-h6fMJm/output/test.tflite
# pm2 start python3 main.py

# https://drive.google.com/file/d/11mujzVaFqa7R1_lB7q0kVPW22Ol51MPg/view?usp=sharing
# gdown --id 11mujzVaFqa7R1_lB7q0kVPW22Ol51MPg
# https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/main/previous_versions/download_tensorflow-2.3.0-cp37-none-linux_armv7l.sh


# tensorflow/tools/ci_build/ci_build.sh PI-PYTHON37 tensorflow/tools/ci_build/pi/build_raspberry_pi.sh
# sudo docker pull armindocachada/tensorflow2-raspberrypi4:2.3.0-cp35-none-linux_armv7l
# sudo docker run -it --rm -v /home/pi/kbai-server:/home armindocachada/tensorflow2-raspberrypi4:2.3.0-cp35-none-linux_armv7l tflite_convert --saved_model_dir=/home/saved_model/ --output_file=/home/projects/project-fsdfs-h6fMJm/output/test.tflite