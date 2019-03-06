# ObjectTracking based on SiameseFC using Tensorflow




## Introduction
...



## Prerequisite
The main requirements to run SiameseFC can be installed by:

```bash
pip install tensorflow    # For CPU
pip install tensorflow-gpu  # For GPU

pip install opencv-python  # Opencv for processing images
```


## Datasets
For training & testing, I used Pascal VOC datasets (2007 and 2012). 
To prapare tha datasets:
1. Download VOC2007 and VOC2012 datasets. I assume the data is stored in /datasets/
```
$ cd datasets
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
2. Convert the data to Tensorflow records:
```
$ tar -xvf VOCtrainval_11-May-2012.tar
$ tar -xvf VOCtrainval_06-Nov-2007.tar
$ tar -xvf VOCtest_06-Nov-2007.tar
$ python3 ssd_image_to_tf.py
```
The resulted tf records will be stored into tfrecords_test and tfrecords_train folders.

## Configuration
Before running the code, you need to touch the configuration based on your needs. There is a config files in /configuration:
- configuration.py: this file includes the common parameters that are used in training, testing and demo.   


## Demo of SiameseFC
For demo, you can run SiameseFC for object tracking in a video.  
Demo uses the pretrained model that has been stored in /Logs/

To run the demo, use the following command:
```python
# Run demo of SiameseFC for a video
python3 siamesefc_demo.py
```
The demo module has the following 5 steps:
1) Load Siamese inference Model 
2) Load input Video
3) Run Tracker
4) Store bounding boxes
5) Visualization & Evaluation

The Output of demo is the video with tracking bounding boxes. 



## Evaluating (Testing) SiameseFC 
This module evaluates the accuracy of SiameseFC with a pretrained model (stored in /Logs) for a testing dataset. 

To test the SiameseFC, use the following command:
```python
# Run test of SSD
python3 siamesefc_test.py
```
Evaluation module has the following 5 steps:
1) Load Siamese inference Model 
2) Load input Video
3) Run Tracker
4) Store bounding boxes
5) Evaluation




## Training SiameseFC
This module is used to train the SiameseFC model for a given dataset. 

To train the SiameseFC, use the following command:
```python
# Run training of SiameseFC
python3 siamesefc_train.py
```

The Training module has the following 2 steps:
1) Create model (training and validation)
2) Training




# How SiameseFC works?
...


![Alt text](figs/siamesefc.jpg?raw=true "SiameseFC")
