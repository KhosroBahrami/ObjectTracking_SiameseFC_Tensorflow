
# configurations for training and tracking
import os.path as osp
import tensorflow as tf
slim = tf.contrib.slim


# Demo
tf.app.flags.DEFINE_string('video_name', 'KiteSurf', 'input video file for demo')  
tf.app.flags.DEFINE_string('demo_video_dir', 'demo_input_videos/', '...')
tf.app.flags.DEFINE_string('run_name_pretrained', 'SiamFC-3s-color-scratch', '...')  
tf.app.flags.DEFINE_string('demo_checkpoint', 'Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained',
                           '...')

# Model 
tf.app.flags.DEFINE_integer('model_z_image_size', 127, 'Exemplar image size')
tf.app.flags.DEFINE_string('model_embedding_name', 'alexnet', '...')
tf.app.flags.DEFINE_string('model_checkpoint_file', None, '...')
tf.app.flags.DEFINE_boolean('model_train_embedding', True, '...')
tf.app.flags.DEFINE_string('model_init_method', 'kaiming_normal', '...')
tf.app.flags.DEFINE_boolean('model_use_bn', True, '...')
tf.app.flags.DEFINE_boolean('model_bn_scale', True, '...')
tf.app.flags.DEFINE_float('model_bn_momentum', 0.05, '...')
tf.app.flags.DEFINE_float('model_bn_epsilon', 1e-6, '...')
tf.app.flags.DEFINE_integer('model_embedding_feature_num', 256, '...')
tf.app.flags.DEFINE_float('model_weight_decay', 5e-4, '...')
tf.app.flags.DEFINE_integer('model_stride', 8, '...')
tf.app.flags.DEFINE_boolean('model_adjust_train_bias', True, '...')
tf.app.flags.DEFINE_float('model_adjust_scale', 1e-3, '...')

# Train
tf.app.flags.DEFINE_string('train_train_dir', 'Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-scratch', '...')
tf.app.flags.DEFINE_integer('train_seed', 123, '...')
tf.app.flags.DEFINE_string('train_input_imdb', 'data/train_imdb.pickle', '...')
tf.app.flags.DEFINE_string('train_preprocessing_name', 'siamese_fc_color', '...')
tf.app.flags.DEFINE_float('train_num_examples_per_epoch', 5.32e4, '...')
tf.app.flags.DEFINE_integer('train_epoch', 50, '...')
tf.app.flags.DEFINE_integer('train_batch_size', 8, '...')
tf.app.flags.DEFINE_integer('train_max_frame_dist', 100, '...')
tf.app.flags.DEFINE_integer('train_prefetch_threads', 4, '...')
tf.app.flags.DEFINE_integer('train_prefetch_capacity', 15 * 8, '...')
tf.app.flags.DEFINE_integer('train_rPos', 16, '...')
tf.app.flags.DEFINE_integer('train_rNeg', 0, '...')
tf.app.flags.DEFINE_string('train_optimizer', 'MOMENTUM', '...')
tf.app.flags.DEFINE_float('train_momentum', 0.9, '...')
tf.app.flags.DEFINE_boolean('train_use_nesterov', False, '...')
tf.app.flags.DEFINE_string('train_policy', 'exponential', '...')
tf.app.flags.DEFINE_float('train_initial_lr', 0.01, '...')
tf.app.flags.DEFINE_integer('train_num_epochs_per_decay', 1, '...')
tf.app.flags.DEFINE_float('train_lr_decay_factor', 0.8685113737513527, '...')
tf.app.flags.DEFINE_boolean('train_staircase', True, '...')
tf.app.flags.DEFINE_integer('train_clip_gradients', None, '...')
tf.app.flags.DEFINE_integer('train_log_every_n_steps', 10, '...')
tf.app.flags.DEFINE_float('train_save_model_every_n_step', 5.32e4 // 8, '...')
tf.app.flags.DEFINE_integer('train_max_checkpoints_to_keep', None, '...')

# Validation
tf.app.flags.DEFINE_string('validation_input_imdb', 'data/validation_imdb.pickle', '...')
tf.app.flags.DEFINE_string('validation_preprocessing_name', 'None', '...')
tf.app.flags.DEFINE_integer('validation_batch_size', 8, '...')
tf.app.flags.DEFINE_integer('validation_max_frame_dist', 100, '...')
tf.app.flags.DEFINE_integer('validation_prefetch_threads', 2, '...')
tf.app.flags.DEFINE_integer('validation_prefetch_capacity', 15 * 8, '...')

# Tracking
tf.app.flags.DEFINE_string('track_log_dir', 'Logs/SiamFC/track_model_inference/SiamFC-3s-color-scratch', '...')
tf.app.flags.DEFINE_integer('track_log_level', 0, '...')
tf.app.flags.DEFINE_integer('track_x_image_size', 255, '...')
tf.app.flags.DEFINE_string('track_upsample_method', 'bicubic', '...')
tf.app.flags.DEFINE_integer('track_upsample_factor', 16, '...')
tf.app.flags.DEFINE_integer('track_num_scales', 3, '...')
tf.app.flags.DEFINE_float('track_scale_step', 1.0375, '...')
tf.app.flags.DEFINE_float('track_scale_damp', 0.59, '...')
tf.app.flags.DEFINE_float('track_scale_penalty', 0.9745, '...')
tf.app.flags.DEFINE_float('track_window_influence', 0.176, '...')
tf.app.flags.DEFINE_boolean('track_include_first', False, '...')



FLAGS = tf.app.flags.FLAGS







