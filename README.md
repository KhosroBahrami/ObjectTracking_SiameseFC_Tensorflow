# Object Tracking based on SiameseFC using Tensorflow

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
For training & testing, I used ILSVRC2015 dataset. 
To prepare the datasets:
1. Download VOC2007 and VOC2012 datasets from http://image-net.org/challenges/LSVRC/2015/
2. Copy the data into /ILSVRC2015 folder



## Configuration
Before running the code, you need to touch the configuration based on your needs. There is a config files in /configuration:
- configuration.py: this file includes the parameters that are used in training, testing and demo.   


## Demo of SiameseFC
For demo, you can run SiameseFC for object tracking in a video.  
Demo uses the pretrained model that has been stored in /Logs/

To run the demo, use the following command:
```python
# Run demo of SiameseFC for a video
python3 siamesefc_demo.py
```
The demo module has the following 5 steps:
1) Load siamese inference model 
2) Load input video
3) Run tracker
4) Store bounding boxes
5) Visualization & evaluation

The output of demo is the video with tracking bounding boxes. 


## Evaluating (Testing) SiameseFC 
This module evaluates the accuracy of SiameseFC with a pretrained model (stored in /Logs) for a testing dataset. 

To test the SiameseFC, use the following command:
```python
# Run test of SiameseFC for a video
python3 siamesefc_test.py
```
Evaluation module has the following 5 steps:
1) Load siamese inference model 
2) Load input video
3) Run tracker
4) Store bounding boxes
5) Evaluation


## Training SiameseFC
This module is used to train the SiameseFC model for a given dataset. 

To train the SiameseFC, use the following command:
```python
# Run training of SiameseFC
python3 siamesefc_train.py
```

The training module has the following 2 steps:
1) Create model (training and validation)
2) Training




# How SiameseFC works?
SiameseFC is a tracking algorithm with a novel fully-convolutional Siamese network trained end-to-end on the ILSVRC15 dataset for object detection in video. It operates at frame-rates beyond real-time and, despite its extreme simplicity, achieves state-of-the-art performance in multiple benchmarks. To tack an object in a video, the object should be identified by a rectangle in the first frame. The following figure shows the concept of Siamese fully convolutional network.

By having the location of the object in the first frame of video, to find the position of the object in the next frame, we can then exhaustively test all possible locations and choose the candidate with the maximum similarity to the appearance of the object in the previous frame. In experiments, the initial appearance of the object is used as the exemplar. 

<!--
![Alt text](figs/siamesefc.jpg?raw=true "SiameseFC")
-->

<p align="center">
  <img  src="figs/siamesefc.jpg" alt="alt text" width="70%" height="70%" title="Siamese network for object tracking">
</p>


### Training
In the training of the siamese network, the goal is to learn a function f (a deep convolutional network) from a dataset of videos with labelled object trajectories. Similarity learning with deep conv-nets is typically addressed using Siamese architectures. Siamese networks apply an identical transformation φ to both inputs and then combine their representations using another function g according to f (z, x) = g(φ(z), φ(x)). When the function g is a simple distance or similarity metric, the function φ can be considered an embedding. Deep Siamese conv-nets have previously been applied to tasks such as face verification, keypoint descriptor learning and one-shot character recognition. 

The advantage of a fully-convolutional network is that, instead of a candidate image of the same size, we can provide as input to the network a much larger search image and it will compute the similarity at all translated sub-windows on a dense grid in a single evaluation. 
To achieve this, we use a convolutional embedding function φ and combine the resulting feature maps using a cross- correlation layer.

The output of this network is not a single score but rather a score map defined on a finite grid. the output of the embedding function is a feature map with spatial support as opposed to a plain vector. 


### Tracking
During tracking, we use a search image centered at the object in the previous frame. The position of the maximum score relative to the center of the score map, multiplied by the stride of the network, gives the displacement of the target from frame to frame. Multiple scales are searched in a single forward-pass by assembling a mini-batch of scaled images. 
Combining feature maps using cross-correlation and evaluating the network once on the larger search image is mathematically equivalent to combining feature maps using the inner product and evaluating the network on each translated sub-window independently. However, the cross-correlation layer provides an incredibly simple method to implement this operation efficiently within the framework of existing conv-net libraries. 




