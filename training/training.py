# Train the model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import os.path as osp
import random
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from configuration.configuration import *



class Training(object):


    def __init__(self):
          a=1

    def _configure_learning_rate(self, global_step):
      num_batches_per_epoch = int(FLAGS.train_num_examples_per_epoch / FLAGS.train_batch_size)
      decay_steps = int(num_batches_per_epoch) * FLAGS.train_num_epochs_per_decay
      return tf.train.exponential_decay(FLAGS.train_initial_lr, global_step, decay_steps=decay_steps,
                                            decay_rate=FLAGS.train_lr_decay_factor, staircase=FLAGS.train_staircase)

 

    def _configure_optimizer(self, learning_rate):
      optimizer_name = FLAGS.train_optimizer.upper()
      if optimizer_name == 'MOMENTUM':
          optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.train_momentum,
            use_nesterov=FLAGS.train_use_nesterov, name='Momentum')
      elif optimizer_name == 'SGD':
          optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      else:
          raise ValueError('Optimizer [%s] was not recognized', optimizer_config['optimizer'])
      return optimizer









     

