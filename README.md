
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
  <img src="https://github.com/mchayan/custom_image_classification_in_ubuntu/blob/master/documentation/1.jpg">
  
Now type this comand to give python path in both research and models directory otherwise you will get error.

``` bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

If everything is perfectly done you will be able to detect object using tensorflow's pre trained model. Type the command below

``` bash
jupyter notebook
```
You will see object_detection_tutorial.ipnyb  file in your browser.

<img src="https://github.com/mchayan/custom_image_classification_in_ubuntu/blob/master/documentation/2.png">
  
Click run-all and you will see output like this 

<img src="https://github.com/mchayan/custom_image_classification_in_ubuntu/blob/master/documentation/3.png">

As it's all done it's time to develop custom object detection

First you need to collect your dataset. Then annotate the images using [Image Labeling Software Like- labelImg](https://github.com/tzutalin/labelImg).
After creating .xml (Annotated File) now it's time to convert the xml files to csv. Create a folder named object-detection. Create folders named data, images, training under object-detection folder.
Create another two folders named test and train inside images folder. Copy all your .xml and image files inside images folder. Copy 80% of your images + perspectiv .xml to train directory and rest of the 20% images + perspectiv .xml in test directory.  

Now run the python file xml_csv.py. The code is available in object-detection Folder.

<img src="https://github.com/mchayan/custom_image_classification_in_ubuntu/blob/master/documentation/4.jpeg">

Now go to research directory and type the command-

``` bash
python3 setup.py install
```

It's time to generate .record files. Go to object-detection directory and run the python file generate_tfrecord.py for both test and train images.

``` bash
python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=./images 
```

``` bash
python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=./images 
```

These will create .record files in data directory.

<img src="https://github.com/mchayan/custom_image_classification_in_ubuntu/blob/master/documentation/5.jpeg">

Now it's time to customize the model. There are many models which can be downloaded. Please visit the link [tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) to download the model you want. We are going to download the ssd_mobilenet_v1_coco_11_06_2017 model  and [ssd_mobilenet_v1_pets.config]
(https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config).
You have to keep the file inside training folder after customization.

We have to make some changes in the file ssd_mobilenet_v1_pets.config . Please open it with any text editor and set the number of the classes = what ever you have labelled.

``` bash
model {
ssd {
num_classes: 2 # we have used two classes so the num classes is 2.  
```
GO to line where you find this code

``` bash
fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"
```
change this line to

``` bash
fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt" #replace PATH_TO_BE_CONFIGURED with the name of the model you have downloaded.
```

Now point to the following lines

``` bash
train_input_reader: {
tf_record_input_reader {
input_path: "PATH_TO_BE_CONFIGURED/pet_faces_train.record-?????-of-00010"
}
label_map_path: "PATH_TO_BE_CONFIGURED/pet_label_map.pbtxt"
}
```

Replace these lines with the following lines

``` bash
train_input_reader: {
tf_record_input_reader {
input_path: "data/train.record"
}
label_map_path: "training/object-detection.pbtxt"
}
```

Now Point to the next lines of the code as below

``` bash
tf_record_input_reader {
input_path: "PATH_TO_BE_CONFIGURED/pet_faces_val.record-?????-of-00010"
}
label_map_path: "PATH_TO_BE_CONFIGURED/pet_label_map.pbtxt"
```

Replace this with the following code

``` bash
tf_record_input_reader {
input_path: "data/test.record"
}
label_map_path: "training/object-detection.pbtxt"
```

Now please create a file object-detection.pbtxt inside the directory training which we have created and write the following lines-

``` bash
# As we have two classes we have included two items in object-detection.pbtxt
item {
  id: 1
  name: 'Unknown Person'
}

item {
  id: 2
  name: 'Known Person'
}
```
Now copy the directories images, data, training and ssd_mobilenet_v1_coco_11_06_2017 inside the directory models/research/object_detection

<img src="https://github.com/mchayan/custom_image_classification_in_ubuntu/blob/master/documentation/6.jpeg">

<img src="https://github.com/mchayan/custom_image_classification_in_ubuntu/blob/master/documentation/7.jpeg">

Not it's time to train. Set the python directory in both research and object_detection diectory each time you encounter any error. 

<img src="https://github.com/mchayan/custom_image_classification_in_ubuntu/blob/master/documentation/8.jpeg">


Now run the command 

``` bash
python3 legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```
<img src="https://github.com/mchayan/custom_image_classification_in_ubuntu/blob/master/documentation/9.jpeg">

Now you can also see the process in the tensorboard. Just open another terminal and issue the below command in the directory models/reseach/object_detection

``` bash
tensorboard --logdir='training'
```
