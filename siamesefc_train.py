
#  Training of object tracking in videos using Siamese Network

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
from sacred.observers import FileStorageObserver
from training import siamese_model
from training.training import *
from configuration.configuration import *


# main module for training of SiameseFC
def main(): 

  with tf.Graph().as_default():


      # 1) Create model (training and validation)
      print('\n1) Create siamese inference model...')
      random.seed(FLAGS.train_seed) 
      np.random.seed(FLAGS.train_seed) 
      tf.set_random_seed(FLAGS.train_seed) 
      model = siamese_model.SiameseModel(mode='train')
      model.build()


      # 2) Training
      print('\n2) Training the model...')
      oTraining = Training() 
      learning_rate = oTraining._configure_learning_rate(model.global_step)
      optimizer = oTraining._configure_optimizer(learning_rate)
      tf.summary.scalar('learning_rate', learning_rate)

      opt_op = tf.contrib.layers.optimize_loss(
          loss=model.total_loss, global_step=model.global_step, learning_rate=learning_rate,
          optimizer=optimizer, clip_gradients=FLAGS.train_clip_gradients, 
          learning_rate_decay_fn=None, summaries=['learning_rate'])

      with tf.control_dependencies([opt_op]):
          train_op = tf.no_op(name='train')

      saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.train_max_checkpoints_to_keep)

      summary_writer = tf.summary.FileWriter(FLAGS.train_train_dir ) 
      summary_op = tf.summary.merge_all()

      global_variables_init_op = tf.global_variables_initializer()
      local_variables_init_op = tf.local_variables_initializer()

      sess = tf.Session()
      model_path = tf.train.latest_checkpoint(FLAGS.train_train_dir)

      if not model_path:
          sess.run(global_variables_init_op)
          sess.run(local_variables_init_op)
          start_step = 0

          if FLAGS.model_embedding_checkpoint_file:
              model.init_fn(sess)
      else:
          print('Restore from last checkpoint: {}'.format(model_path))
          sess.run(local_variables_init_op)
          saver.restore(sess, model_path)
          start_step = tf.train.global_step(sess, model.global_step.name) + 1


      total_steps = int(FLAGS.train_epoch * FLAGS.train_num_examples_per_epoch / FLAGS.train_batch_size)
      print('Train for {} steps'.format(total_steps))
      for step in range(start_step, total_steps):
          start_time = time.time()
          _, loss, batch_loss = sess.run([train_op, model.total_loss, model.batch_loss])
          duration = time.time() - start_time

          if step % 10 == 0:
              examples_per_sec = FLAGS.train_batch_size / float(duration)
              time_remain = FLAGS.train_batch_size * (total_steps - step) / examples_per_sec
              m, s = divmod(time_remain, 60)
              h, m = divmod(m, 60)
              format_str = ('%s: step %d, total loss = %.2f, batch loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch; %dh:%02dm:%02ds remains)')
              print(format_str % (datetime.now(), step, loss, batch_loss, examples_per_sec, duration, h, m, s))
           
          if step % 100 == 0:
              summary_str = sess.run(summary_op)
              summary_writer.add_summary(summary_str, step)
          
          if step % FLAGS.train_save_model_every_n_step == 0 or (step + 1) == total_steps:
              checkpoint_path = osp.join(FLAGS.train_train_dir, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=step)




if __name__ == '__main__':
    main()




