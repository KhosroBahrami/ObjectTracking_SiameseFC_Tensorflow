
# Siamese Class for inference 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import logging
import os
import os.path as osp
import numpy as np
import tensorflow as tf
from networks.alexnet import convolutional_alexnet_arg_scope, convolutional_alexnet
from inference.infer_utils import get_exemplar_images, get_center, get, convert_bbox_format, Rectangle
import cv2
from cv2 import imwrite
from configuration.configuration import *

slim = tf.contrib.slim



# Represent the target state.
class TargetState(object):

  def __init__(self, bbox, search_pos, scale_idx):
      self.bbox = bbox  # (cx, cy, w, h) in the original image
      self.search_pos = search_pos  # target center position in the search image
      self.scale_idx = scale_idx  # scale index in the searched scales



# Model wrapper class for performing inference with a siamese model
class SiameseInference():

  def __init__(self): #, model_config, track_config):
      self.image = None
      self.target_bbox_feed = None
      self.search_images = None
      self.embeds = None
      self.templates = None
      self.init = None
      self.response_up = None

      self.num_scales = FLAGS.track_num_scales 
      logging.info('track num scales -- {}'.format(self.num_scales))
      scales = np.arange(self.num_scales) - get_center(self.num_scales)
      self.search_factors = [FLAGS.track_scale_step ** x for x in scales]

      self.x_image_size = FLAGS.track_x_image_size  # Search image size
      self.window = None  # Cosine window
      self.log_level = FLAGS.track_log_level 



  # Load siamese model into session
  def load_model(self, sess, checkpoint_path):
      self.build_inputs()
      self.build_search_images()
      self.build_template()
      self.build_detection()
      self.build_upsample()
      self.dumb_op = tf.no_op('dumb_operation')
      ema = tf.train.ExponentialMovingAverage(0)
      variables_to_restore = ema.variables_to_restore(moving_avg_variables=[])
      # Filter out State variables
      variables_to_restore_filterd = {}
      for key, value in variables_to_restore.items():
         if key.split('/')[1] != 'State':
            variables_to_restore_filterd[key] = value
      saver = tf.train.Saver(variables_to_restore_filterd)
      if osp.isdir(checkpoint_path):
         checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
      saver.restore(sess, checkpoint_path)




  def build_inputs(self):
      filename = tf.placeholder(tf.string, [], name='filename')
      image_file = tf.read_file(filename)
      image = tf.image.decode_jpeg(image_file, channels=3, dct_method="INTEGER_ACCURATE")
      image = tf.to_float(image)
      self.image = image
      self.target_bbox_feed = tf.placeholder(dtype=tf.float32, shape=[4], name='target_bbox_feed')  



  #Crop search images from the input image based on the last target position
  #  1. The input image is scaled such that the area of target&context takes up to (scale_factor * z_image_size) ^ 2
  #  2. Crop an image patch as large as x_image_size centered at the target center.
  #  3. If the cropped image region is beyond the boundary of the input image, mean values are padded.
  def build_search_images(self):
      size_z = FLAGS.model_z_image_size  
      size_x = FLAGS.track_x_image_size  
      context_amount = 0.5

      num_scales = FLAGS.track_num_scales 
      scales = np.arange(num_scales) - get_center(num_scales)
      assert np.sum(scales) == 0, 'scales should be symmetric'
      search_factors = [FLAGS.track_scale_step ** x for x in scales]

      frame_sz = tf.shape(self.image)
      target_yx = self.target_bbox_feed[0:2]
      target_size = self.target_bbox_feed[2:4]
      avg_chan = tf.reduce_mean(self.image, axis=(0, 1), name='avg_chan')

      # Compute base values
      base_z_size = target_size
      base_z_context_size = base_z_size + context_amount * tf.reduce_sum(base_z_size)
      base_s_z = tf.sqrt(tf.reduce_prod(base_z_context_size))  # Canonical size
      base_scale_z = tf.div(tf.to_float(size_z), base_s_z)
      d_search = (size_x - size_z) / 2.0
      base_pad = tf.div(d_search, base_scale_z)
      base_s_x = base_s_z + 2 * base_pad
      base_scale_x = tf.div(tf.to_float(size_x), base_s_x)

      boxes = []
      for factor in search_factors:
          s_x = factor * base_s_x
          frame_sz_1 = tf.to_float(frame_sz[0:2] - 1)
          topleft = tf.div(target_yx - get_center(s_x), frame_sz_1)
          bottomright = tf.div(target_yx + get_center(s_x), frame_sz_1)
          box = tf.concat([topleft, bottomright], axis=0)
          boxes.append(box)
      boxes = tf.stack(boxes)

      scale_xs = []
      for factor in search_factors:
          scale_x = base_scale_x / factor
          scale_xs.append(scale_x)
      self.scale_xs = tf.stack(scale_xs)

      # Note we use different padding values for each image
      # while the original implementation uses only the average value
      # of the first image for all images.
      image_minus_avg = tf.expand_dims(self.image - avg_chan, 0)
      image_cropped = tf.image.crop_and_resize(image_minus_avg, boxes,
                                               box_ind=tf.zeros((FLAGS.track_num_scales), tf.int32),
                                               crop_size=[size_x, size_x])
      self.search_images = image_cropped + avg_chan



  def get_image_embedding(self, images, reuse=None):
      arg_scope = convolutional_alexnet_arg_scope(trainable=FLAGS.model_train_embedding, is_training=False)
      @functools.wraps(convolutional_alexnet)
      def embedding_fn(images, reuse=False):
        with slim.arg_scope(arg_scope):
          return convolutional_alexnet(images, reuse=reuse)
      embed, _ = embedding_fn(images, reuse)
      return embed



  def build_template(self):
      exemplar_images = get_exemplar_images(self.search_images, [FLAGS.model_z_image_size,
                                                                 FLAGS.model_z_image_size])
      
      templates = self.get_image_embedding(exemplar_images)
      center_scale = int(get_center(FLAGS.track_num_scales))
      center_template = tf.identity(templates[center_scale])
      templates = tf.stack([center_template for _ in range(FLAGS.track_num_scales)])

      with tf.variable_scope('target_template'):
          # Store template in Variable such that we don't have to feed this template every time.
          with tf.variable_scope('State'):
              state = tf.get_variable('exemplar',
                                      initializer=tf.zeros(templates.get_shape().as_list(),
                                      dtype=templates.dtype),
                                      trainable=False)
              with tf.control_dependencies([templates]):
                  self.init = tf.assign(state, templates, validate_shape=True)
              self.templates = state



  def build_detection(self):
      self.embeds = self.get_image_embedding(self.search_images, reuse=True)
      with tf.variable_scope('detection'):
          def _translation_match(x, z):
              x = tf.expand_dims(x, 0)  # [batch, in_height, in_width, in_channels]
              z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, out_channels]
              return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')

          output = tf.map_fn(lambda x: _translation_match(x[0], x[1]),
            (self.embeds, self.templates), dtype=self.embeds.dtype)  # of shape [16, 1, 17, 17, 1]
          output = tf.squeeze(output, [1, 4])  # of shape e.g. [16, 17, 17]

          bias = tf.get_variable('biases', [1], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32), trainable=False)
          response = FLAGS.model_adjust_scale * output + bias
          self.response = response



  # Upsample response to obtain finer target position
  def build_upsample(self):
      with tf.variable_scope('upsample'):
        response = tf.expand_dims(self.response, 3)
        up_method = FLAGS.track_upsample_method
        methods = {'bilinear': tf.image.ResizeMethod.BILINEAR, 'bicubic': tf.image.ResizeMethod.BICUBIC}
        up_method = methods[up_method]
        response_spatial_size = self.response.get_shape().as_list()[1:3]
        up_size = [s * FLAGS.track_upsample_factor for s in response_spatial_size]
        response_up = tf.image.resize_images(response, up_size, method=up_method, align_corners=True)
        response_up = tf.squeeze(response_up, [3])
        self.response_up = response_up



  def initialize(self, sess, input_feed):
      image_path, target_bbox = input_feed
      scale_xs, _ = sess.run([self.scale_xs, self.init], feed_dict={'filename:0': image_path,
                                        "target_bbox_feed:0": target_bbox, })
      return scale_xs

  
  def inference_step(self, sess, input_feed):
      image_path, target_bbox = input_feed
      log_level = FLAGS.track_log_level
      image_cropped_op = self.search_images if log_level > 0 else self.dumb_op
      image_cropped, scale_xs, response_output = sess.run(
        fetches=[image_cropped_op, self.scale_xs, self.response_up],
        feed_dict={"filename:0": image_path, "target_bbox_feed:0": target_bbox, })

      output = {'image_cropped': image_cropped, 'scale_xs': scale_xs, 'response': response_output}
      return output, None



  # Runs tracking on a single image sequence.
  def track(self, sess, first_bbox, frames, logdir='/tmp'):
      # Get initial target bounding box and convert to center based
      bbox = convert_bbox_format(first_bbox, 'center-based')
      # Feed in the first frame image to set initial state.
      bbox_feed = [bbox.y, bbox.x, bbox.height, bbox.width]
      input_feed = [frames[0], bbox_feed]
      frame2crop_scale = self.initialize(sess, input_feed)
      # Storing target state
      original_target_height = bbox.height
      original_target_width = bbox.width
      search_center = np.array([get_center(self.x_image_size), get_center(self.x_image_size)])
      current_target_state = TargetState(bbox=bbox, search_pos=search_center,
                                         scale_idx=int(get_center(self.num_scales)))

      include_first = get(FLAGS.track_include_first, False)
      logging.info('Tracking include first -- {}'.format(include_first))

      # Run tracking loop
      reported_bboxs = []
      for i, filename in enumerate(frames):
          if i > 0 or include_first:  # We don't really want to process the first image unless intended to do so.
            bbox_feed = [current_target_state.bbox.y, current_target_state.bbox.x,
                         current_target_state.bbox.height, current_target_state.bbox.width]
            input_feed = [filename, bbox_feed]

            outputs, metadata = self.inference_step(sess, input_feed)
            search_scale_list = outputs['scale_xs']
            response = outputs['response']
            response_size = response.shape[1]

            # Choose the scale whole response map has the highest peak
            if self.num_scales > 1:
                response_max = np.max(response, axis=(1, 2))
                penalties = FLAGS.track_scale_penalty * np.ones((self.num_scales))
                current_scale_idx = int(get_center(self.num_scales))
                penalties[current_scale_idx] = 1.0
                response_penalized = response_max * penalties
                best_scale = np.argmax(response_penalized)
            else:
                best_scale = 0

            response = response[best_scale]

            with np.errstate(all='raise'):  # Raise error if something goes wrong
                response = response - np.min(response)
                response = response / np.sum(response)

            if self.window is None:
                window = np.dot(np.expand_dims(np.hanning(response_size), 1),
                                np.expand_dims(np.hanning(response_size), 0))
                self.window = window / np.sum(window)  # normalize window
            window_influence = FLAGS.track_window_influence
            response = (1 - window_influence) * response + window_influence * self.window

            # Find maximum response
            r_max, c_max = np.unravel_index(response.argmax(), response.shape)
            # Convert from crop-relative coordinates to frame coordinates
            p_coor = np.array([r_max, c_max])
            # displacement from the center in instance final representation ...
            disp_instance_final = p_coor - get_center(response_size)
            # ... in instance feature space ...
            upsample_factor = FLAGS.track_upsample_factor
            disp_instance_feat = disp_instance_final / upsample_factor
            # ... Avoid empty position ...
            r_radius = int(response_size / upsample_factor / 2)
            disp_instance_feat = np.maximum(np.minimum(disp_instance_feat, r_radius), -r_radius)
            # ... in instance input ...
            disp_instance_input = disp_instance_feat * FLAGS.model_stride
            # ... in instance original crop (in frame coordinates)
            disp_instance_frame = disp_instance_input / search_scale_list[best_scale]
            # Position within frame in frame coordinates
            y = current_target_state.bbox.y
            x = current_target_state.bbox.x
            y += disp_instance_frame[0]
            x += disp_instance_frame[1]
            # Target scale damping and saturation
            target_scale = current_target_state.bbox.height / original_target_height
            search_factor = self.search_factors[best_scale]
            scale_damp = FLAGS.track_scale_damp  # damping factor for scale update
            target_scale *= ((1 - scale_damp) * 1.0 + scale_damp * search_factor)
            target_scale = np.maximum(0.2, np.minimum(5.0, target_scale))
            # Some book keeping
            height = original_target_height * target_scale
            width = original_target_width * target_scale
            current_target_state.bbox = Rectangle(x, y, width, height)
            current_target_state.scale_idx = best_scale
            current_target_state.search_pos = search_center + disp_instance_input

            assert 0 <= current_target_state.search_pos[0] < self.x_image_size, \
              'target position in feature space should be no larger than input image size'
            assert 0 <= current_target_state.search_pos[1] < self.x_image_size, \
              'target position in feature space should be no larger than input image size'

            if self.log_level > 0:
                np.save(osp.join(logdir, 'num_frames.npy'), [i + 1])
                # Select the image with the highest score scale and convert it to uint8
                image_cropped = outputs['image_cropped'][best_scale].astype(np.uint8)
                # Note that imwrite in cv2 assumes the image is in BGR format.
                # However, the cropped image returned by TensorFlow is RGB.
                # Therefore, we convert color format using cv2.cvtColor
                imwrite(osp.join(logdir, 'image_cropped{}.jpg'.format(i)), cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR))
                np.save(osp.join(logdir, 'best_scale{}.npy'.format(i)), [best_scale])
                np.save(osp.join(logdir, 'response{}.npy'.format(i)), response)

                y_search, x_search = current_target_state.search_pos
                search_scale = search_scale_list[best_scale]
                target_height_search = height * search_scale
                target_width_search = width * search_scale
                bbox_search = Rectangle(x_search, y_search, target_width_search, target_height_search)
                bbox_search = convert_bbox_format(bbox_search, 'top-left-based')
                np.save(osp.join(logdir, 'bbox{}.npy'.format(i)),
                        [bbox_search.x, bbox_search.y, bbox_search.width, bbox_search.height])

          reported_bbox = convert_bbox_format(current_target_state.bbox, 'top-left-based')
          reported_bboxs.append(reported_bbox)
      return reported_bboxs






