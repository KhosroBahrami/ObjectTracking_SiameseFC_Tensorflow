
# Test of object tracking in videos using Siamese Network
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import os.path as osp
import sys
from glob import glob
import tensorflow as tf
from sacred import Experiment
from inference.infer_utils import Rectangle, mkdir_p, sort_nicely
from inference.siamese_inference import *
from evaluation.evaluation import * 
from configuration.configuration import *
from datasets.dataset import DataSet


# main module for testing
def main():

    with tf.Session() as sess:


        # 1) Load Siamese inference Model
        print('\n1) Load siamese inference model...')
        oSiameseInference = SiameseInference() 
        oSiameseInference.load_model(sess, FLAGS.demo_checkpoint)
        print('\nmodel :',oSiameseInference)

        print(FLAGS.demo_video_dir)

        subfolders = [f.path for f in os.scandir(FLAGS.demo_video_dir) if f.is_dir() ] 
        
        for demo_video_dir in subfolders:
          
        
            # 2) Load input Video
            print('\n2) Load input Video...')
            video_name = osp.basename(demo_video_dir)
            video_log_dir = osp.join(FLAGS.track_log_dir, video_name)
            mkdir_p(video_log_dir)
            filenames = sort_nicely(glob(demo_video_dir + '/img/*.jpg'))
            first_line = open(demo_video_dir + '/groundtruth_rect.txt').readline()
            bbox = [int(v) for v in first_line.strip().split(',')]
            initial_bbox = Rectangle(bbox[0] - 1, bbox[1] - 1, bbox[2], bbox[3]) 
            print('input video directory: ', demo_video_dir)
            print('video_log_dir: ',video_log_dir)
            print('initial boundingbox: ',initial_bbox)


            # 3) Run Tracker
            print('\n3) Run Tracker...')
            trajectory = oSiameseInference.track(sess, initial_bbox, filenames, video_log_dir)
            print('trajectory: ',trajectory)


            # 4) Store bounding boxes
            print('\n4) Store bounding boxes...')
            with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
                for region in trajectory:
                   rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1, region.width, region.height)
                   f.write(rect_str)


            # 5) Evaluation 
            with tf.name_scope(None, "Evaluation") as scope:
               print('\n5) Evaluation...')
               oEvaluation = Evaluation()
               oEvaluation.calculate_accuracy(filenames) 
        



if __name__ == '__main__':
    main()













                  
