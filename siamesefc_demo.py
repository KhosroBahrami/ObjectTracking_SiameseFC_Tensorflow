
#  Demo of object tracking in videos using Siamese Network
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import os.path as osp
import sys
from glob import glob
import tensorflow as tf
from inference.infer_utils import Rectangle, mkdir_p, sort_nicely
from inference.siamese_inference import *
from visualization.visualization import * 
from configuration.configuration import *


# main module for demo
def main():

    with tf.Session() as sess:

        # 1) Load Siamese inference Model
        print('\n1) Load siamese inference model...')
        oSiameseInference = SiameseInference() 
        oSiameseInference.load_model(sess, FLAGS.demo_checkpoint)
        print('\nmodel :',oSiameseInference)

        
        # 2) Load input Video
        print('\n2) Load input Video...')
        video_log_dir = osp.join(FLAGS.track_log_dir, FLAGS.video_name)
        mkdir_p(video_log_dir)
        filenames = sort_nicely(glob(FLAGS.demo_video_dir + FLAGS.video_name + '/img/*.jpg'))
        first_line = open(FLAGS.demo_video_dir + FLAGS.video_name + '/groundtruth_rect.txt').readline()
        bbox = [int(v) for v in first_line.strip().split(',')]
        initial_bbox = Rectangle(bbox[0] - 1, bbox[1] - 1, bbox[2], bbox[3])  
        print('\ninput video directory: ', FLAGS.demo_video_dir)
        print('\nvideo_log_dir: ',video_log_dir)
        print('\ninitial boundingbox location: ',initial_bbox)


        # 3) Run Tracker
        print('\n3) Run Tracker...')
        trajectory = oSiameseInference.track(sess, initial_bbox, filenames, video_log_dir)


        # 4) Store bounding boxes
        print('\n4) Store bounding boxes...')
        with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
            for region in trajectory:
               rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1, region.width, region.height)
               f.write(rect_str)


        # 5) Visualization 
        with tf.name_scope(None, "Visualization") as scope:
           print('\n5) Visualization...')
           oVisualization = Visualization()
           oVisualization.plot_bboxes(filenames) 





if __name__ == '__main__':
    main()













                  
