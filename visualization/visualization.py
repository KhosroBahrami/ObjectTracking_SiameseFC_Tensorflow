
# Visualization of SiameseFC for demo
import os
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm
from matplotlib import pyplot
import os.path as osp
from configuration.configuration import *


class Visualization(object):

    def __init__(self):      
        a=1


    # Visualize bounding boxes
    def plot_bboxes(self, filenames): 

        track_log_dir = 'Logs/SiamFC/track_model_inference/{}/{}'.format(FLAGS.run_name_pretrained, FLAGS.video_name)
        te_bboxs = self.readbbox(osp.join(track_log_dir, 'track_rect.txt'))
        gt_bboxs = self.readbbox(osp.join(FLAGS.demo_video_dir + FLAGS.video_name, 'groundtruth_rect.txt'))
        print('\nnum_frames :', len(gt_bboxs))
        
        for frame in range(len(filenames)):
            img = mpimg.imread(filenames[frame])
            
            cv2.rectangle(img,(int(te_bboxs[frame][0]),int(te_bboxs[frame][1])),
                              (int(te_bboxs[frame][0])+int(te_bboxs[frame][2]),
                               int(te_bboxs[frame][1])+int(te_bboxs[frame][3])),(200,0,0),2)
            
            cv2.rectangle(img,(int(gt_bboxs[frame][0]),int(gt_bboxs[frame][1])),
                              (int(gt_bboxs[frame][0])+int(gt_bboxs[frame][2]),
                               int(gt_bboxs[frame][1])+int(gt_bboxs[frame][3])),(0,200,0),2)
            
            plt.imsave('demo_output_images/frame_'+str(frame), img)

        # Convert images to video    
        os.system("ffmpeg -r 25 -i demo_output_images/frame_%d.png -vcodec mpeg4 -y demo_output_video/"+FLAGS.video_name+".mp4")
                


    def readbbox(self, file):
      with open(file, 'r') as f:
        lines = f.readlines()
        bboxs = [[float(val) for val in line.strip().replace(' ', ',').replace('\t', ',').split(',')] for line in lines]
      return bboxs




    
