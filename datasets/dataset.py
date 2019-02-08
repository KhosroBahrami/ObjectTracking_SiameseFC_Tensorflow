
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import tensorflow as tf
from datasets.sampler import Sampler
from datasets.preprocessing import Compose, RandomGray, RandomCrop, CenterCrop, RandomStretch
from datasets.vid import VID

from configuration.configuration import *


class DataSet(object):
  def __init__(self, mode, is_training):
    if mode=='train':
       self.config_input_imdb = FLAGS.train_input_imdb 
       self.config_preprocessing_name = FLAGS.train_preprocessing_name
       self.config_batch_size = FLAGS.train_batch_size
       self.config_max_frame_dist = FLAGS.train_max_frame_dist 
       self.config_prefetch_threads = FLAGS.train_prefetch_threads 
       self.config_prefetch_capacity = FLAGS.train_prefetch_capacity 
    else:
       self.config_input_imdb = FLAGS.validation_input_imdb 
       self.config_preprocessing_name = FLAGS.validation_preprocessing_name
       self.config_batch_size = FLAGS.validation_batch_size
       self.config_max_frame_dist = FLAGS.validation_max_frame_dist 
       self.config_prefetch_threads = FLAGS.validation_prefetch_threads 
       self.config_prefetch_capacity = FLAGS.validation_prefetch_capacity 
    
    #self.config = config
    self.is_training = is_training

    preprocess_name = self.config_preprocessing_name 
    logging.info('preproces -- {}'.format(preprocess_name))

    if preprocess_name == 'siamese_fc_color':
      self.v_transform = None
      # TODO: use a single operation (tf.image.crop_and_resize) to achieve all transformations ?
      self.z_transform = Compose([RandomStretch(),
                                  CenterCrop((255 - 8, 255 - 8)),
                                  RandomCrop(255 - 2 * 8),
                                  CenterCrop((127, 127))])
      self.x_transform = Compose([RandomStretch(),
                                  CenterCrop((255 - 8, 255 - 8)),
                                  RandomCrop(255 - 2 * 8), ])
    elif preprocess_name == 'siamese_fc_gray':
      self.v_transform = RandomGray()
      self.z_transform = Compose([RandomStretch(),
                                  CenterCrop((255 - 8, 255 - 8)),
                                  RandomCrop(255 - 2 * 8),
                                  CenterCrop((127, 127))])
      self.x_transform = Compose([RandomStretch(),
                                  CenterCrop((255 - 8, 255 - 8)),
                                  RandomCrop(255 - 2 * 8), ])
    elif preprocess_name == 'None':
      self.v_transform = None
      self.z_transform = CenterCrop((127, 127))
      self.x_transform = CenterCrop((255, 255))
    else:
      raise ValueError('Preprocessing name {} was not recognized.'.format(preprocess_name))

    self.dataset_py = VID(self.config_input_imdb, self.config_max_frame_dist)
    self.sampler = Sampler(self.dataset_py, shuffle=is_training)

  def build(self):
    self.build_dataset()
    self.build_iterator()

  def build_dataset(self):
    def sample_generator():
      for video_id in self.sampler:
        sample = self.dataset_py[video_id]
        yield sample

    def transform_fn(video):
      exemplar_file = tf.read_file(video[0])
      instance_file = tf.read_file(video[1])
      exemplar_image = tf.image.decode_jpeg(exemplar_file, channels=3, dct_method="INTEGER_ACCURATE")
      instance_image = tf.image.decode_jpeg(instance_file, channels=3, dct_method="INTEGER_ACCURATE")

      if self.v_transform is not None:
        video = tf.stack([exemplar_image, instance_image])
        video = self.v_transform(video)
        exemplar_image = video[0]
        instance_image = video[1]

      if self.z_transform is not None:
        exemplar_image = self.z_transform(exemplar_image)

      if self.x_transform is not None:
        instance_image = self.x_transform(instance_image)

      return exemplar_image, instance_image

    dataset = tf.data.Dataset.from_generator(sample_generator,
                                             output_types=(tf.string),
                                             output_shapes=(tf.TensorShape([2])))
    dataset = dataset.map(transform_fn, num_parallel_calls = self.config_prefetch_threads)
    dataset = dataset.prefetch(self.config_prefetch_capacity)
    dataset = dataset.repeat()
    dataset = dataset.batch(self.config_batch_size)
    self.dataset_tf = dataset

  def build_iterator(self):
    self.iterator = self.dataset_tf.make_one_shot_iterator()

  def get_one_batch(self):
    return self.iterator.get_next()



  
