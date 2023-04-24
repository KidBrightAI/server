# install tflite runtime
sudo apt-get install -y cmake curl
wget -O tensorflow.zip https://github.com/tensorflow/tensorflow/archive/v2.4.1.zip
unzip tensorflow.zip
mv tensorflow-2.4.1 tensorflow
cd tensorflow
./tensorflow/lite/tools/make/download_dependencies.sh
./tensorflow/lite/tools/pip_package/build_pip_package.sh
cd ..
rm tensorflow.zip
pip3 install /home/pi/server/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/tflite_runtime-2.4.1-cp36-cp36m-linux_aarch64.whl

# install dep
pip3 install -r requirements_jetson.txt

# install nodejs 
cd ssh
npm install
cd ..

#startup process
# 1. kidbright.service
# cat /lib/systemd/system/kidbright.service
# 2. startup file
# cat /usr/sbin/kidbright-start
# 3. pm2 => run server
# pm2 list
