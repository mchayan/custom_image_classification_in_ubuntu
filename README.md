
<h1>Custom image classification in Ubuntu</h1>
This custom image classification project is developed in ubuntu. Aim of this project is to identify whether the person is wearing glass or not. The final output will classify a person as known person (If the person is wearing a glass) or unknown person (If the person is  not wearing a glass) in any video. Here is the step by step procedure of how to develop it.


Alt-Make Your Environment Ready
 
# Installation

## Dependencies

Tensorflow Object Detection API depends on the following libraries:

*   Protobuf 3.0.0
*   Python-tk
*   Pillow 1.0
*   lxml
*   tf Slim (which is included in the "tensorflow/models/research/" checkout)
*   Jupyter notebook
*   Matplotlib
*   Tensorflow (>=1.12.0)
*   Cython
*   contextlib2
*   cocoapi


Open the terminal and write these command bellow.

``` bash
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```
  
Install dependencies using pip:

``` bash
pip install contextlib2
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib
```

All the required dependencies are easy to install except protoc. To install protoc type the command bellow.

``` bash
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
```

As all the depedecies installation is done, it's time to check the configuration is done or not. To verify please download the office Tensorflow-model from- 

``` bash
git clone https://github.com/tensorflow/models.git
```

Go to the research directory of your downloaded git model and type the command 
``` bash
protoc object_detection/protos/*.proto –python_out=.
```

like this-
  <img src="https://github.com/mchayan/custom_image_classification_in_ubuntu/blob/master/documentation/1.jpeg">
