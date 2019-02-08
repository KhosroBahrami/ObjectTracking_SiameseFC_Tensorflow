
# Evaluation of SiameseFC for testing
import os
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm
from matplotlib import pyplot
import os.path as osp
from configuration.configuration import *

class Evaluation(object):

    def __init__(self):      
        a=1


    # Evaluating bounding boxes
    def calculate_accuracy(self, filenames): 

        runname = 'SiamFC-3s-color-pretrained'
        track_log_dir = 'Logs/SiamFC/track_model_inference/{}/{}'.format(runname, FLAGS.video_name)
        
        track_log_dir = osp.join(track_log_dir)
        video_data_dir = osp.join(FLAGS.demo_video_dir, FLAGS.video_name)
        te_bboxs = self.readbbox(osp.join(track_log_dir, 'track_rect.txt'))
        gt_bboxs = self.readbbox(osp.join(video_data_dir, 'groundtruth_rect.txt'))
        num_frames = len(gt_bboxs)

        print('te_bboxs :',te_bboxs)
        print('gt_bboxs :',gt_bboxs)
        print('num_frames :',num_frames)

        for ind in range(num_frames):
            print(te_bboxs[ind], gt_bboxs[ind])
        
    
       
        


    def readbbox(self, file):
      with open(file, 'r') as f:
        lines = f.readlines()
        bboxs = [[float(val) for val in line.strip().replace(' ', ',').replace('\t', ',').split(',')] for line in lines]
      return bboxs




    
