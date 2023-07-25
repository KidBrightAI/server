sudo apt-get install -y cmake curl
# clean disk 
sudo apt autoremove -y
sudo apt clean
sudo apt remove thunderbird libreoffice-* -y
sudo rm -rf /usr/local/cuda/samples /usr/src/cudnn_samples_* /usr/src/tensorrt/data /usr/src/tensorrt/samples /usr/share/visionworks* ~/VisionWorks-SFM*Samples /opt/nvidia/deepstream/deepstream*/samples


# install tflite runtime
#wget -O tensorflow.zip https://github.com/tensorflow/tensorflow/archive/v2.4.1.zip
#unzip tensorflow.zip
#mv tensorflow-2.4.1 tensorflow
#cd tensorflow
#./tensorflow/lite/tools/make/download_dependencies.sh
#./tensorflow/lite/tools/pip_package/build_pip_package.sh
#cd ..
#rm tensorflow.zip
#pip3 install /home/pi/server/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/tflite_runtime-2.4.1-cp36-cp36m-linux_aarch64.whl

# install dep
pip3 install -r requirements_jetson.txt

#========= install tensorflow ========= 
# remove 1.14.X
sudo pip uninstall tensorflow
sudo pip3 uninstall tensorflow
# install the dependencies (if not already onboard)
sudo apt-get -y install gfortran
sudo apt-get -y install libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get -y install libatlas-base-dev libopenblas-dev libblas-dev
sudo apt-get -y install liblapack-dev
sudo -H pip3 install Cython==0.29.21
# install h5py with Cython version 0.29.21 (± 6 min @1950 MHz)
sudo -H pip3 install h5py==2.10.0
sudo -H pip3 install -U testresources
# upgrade setuptools 39.0.1 -> 53.0.0
sudo -H pip3 install --upgrade setuptools
sudo -H pip3 install pybind11 protobuf google-pasta
sudo -H pip3 install -U six mock wheel requests gast
sudo -H pip3 install keras_applications --no-deps
sudo -H pip3 install keras_preprocessing --no-deps
# install gdown to download from Google drive
sudo -H pip3 install gdown
# download the wheel
gdown https://drive.google.com/uc?id=1DLk4Tjs8Mjg919NkDnYg02zEnbbCAzOz
# install TensorFlow (± 12 min @1500 MHz)
sudo -H pip3 install tensorflow-2.4.1-cp36-cp36m-linux_aarch64.whl

sudo -H pip3 install tensorflowjs

sudo -H pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl#sha256=7c58b1a9fb2d2b24d6f0b0f8629ede7d288358e2cb93c68c3e4f78fd0ee7d1df

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

sudo chmod +x /home/pi/server/scripts/expand_sdcard_jetson.sh
sudo /home/pi/server/scripts/expand_sdcard_jetson.sh