# ObjectTracking based on SiameseFC using Tensorflow

This is a TensorFlow implementation of Fully-Convolutional Siamese Networks for Object Tracking published in [paper](https://arxiv.org/abs/1606.09549). This repository contains a TensorFlow re-implementation of SiameseFC which is inspired by the previous implementations. However, this code has clear pipelines for train, test and demo; it is modular that can be extended or can be used for new applications.

This implementation is designed with the following goals:
- Clear Pipeline: it has full pipeline of object detection for demo, test and train with seperate modules.
- Modularity: This code is modular and easy to expand for any specific application or new ideas.
- To be deployed on Embedded Systems 



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
SiameseFC is a tracking algorithm with a novel fully-convolutional Siamese network trained end-to-end on the ILSVRC15 dataset for object detection in video. It operates at frame-rates beyond real-time and, despite its extreme simplicity, achieves state-of-the-art performance in multiple benchmarks. To tack an object in a video, the object should be identified by a rectangle in the first frame. 

![Alt text](figs/siamesefc.jpg?raw=true "SiameseFC")
