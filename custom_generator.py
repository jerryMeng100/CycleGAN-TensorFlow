from __future__ import division
# CycleGAN imports
import tensorflow as tf
import ops
import utils
import custom_ops
import custom_utils

# pix2pix imports
import os
import time
from glob import glob
# import tensorflow as tf
import numpy as np
from six.moves import xrange

class Generator:
  def __init__(self, name, is_training, ngf=64, norm='instance', image_size=256):
    # tf.reset_default_graph()
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size
    self.output_size = image_size

    self.should_reuse = False

    self.input_c_dim = 3
    self.output_c_dim = 3

    self.batch_size = 1

    self.gf_dim = 64
    self.df_dim = 64

    # pix2pix batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = custom_ops.batch_norm(name='d_bn1')
    self.d_bn2 = custom_ops.batch_norm(name='d_bn2')
    self.d_bn3 = custom_ops.batch_norm(name='d_bn3')

    self.g_bn_e2 = custom_ops.batch_norm(name='g_bn_e2')
    self.g_bn_e3 = custom_ops.batch_norm(name='g_bn_e3')
    self.g_bn_e4 = custom_ops.batch_norm(name='g_bn_e4')
    self.g_bn_e5 = custom_ops.batch_norm(name='g_bn_e5')
    self.g_bn_e6 = custom_ops.batch_norm(name='g_bn_e6')
    self.g_bn_e7 = custom_ops.batch_norm(name='g_bn_e7')
    self.g_bn_e8 = custom_ops.batch_norm(name='g_bn_e8')

    self.g_bn_d1 = custom_ops.batch_norm(name='g_bn_d1')
    self.g_bn_d2 = custom_ops.batch_norm(name='g_bn_d2')
    self.g_bn_d3 = custom_ops.batch_norm(name='g_bn_d3')
    self.g_bn_d4 = custom_ops.batch_norm(name='g_bn_d4')
    self.g_bn_d5 = custom_ops.batch_norm(name='g_bn_d5')
    self.g_bn_d6 = custom_ops.batch_norm(name='g_bn_d6')
    self.g_bn_d7 = custom_ops.batch_norm(name='g_bn_d7')

  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    # with tf.variable_scope(self.name):
    #   # conv layers
    #   c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
    #       reuse=self.reuse, name='c7s1_32')                             # (?, w, h, 32)
    #   d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
    #       reuse=self.reuse, name='d64')                                 # (?, w/2, h/2, 64)
    #   d128 = ops.dk(d64, 4*self.ngf, is_training=self.is_training, norm=self.norm,
    #       reuse=self.reuse, name='d128')                                # (?, w/4, h/4, 128)

    #   if self.image_size <= 128:
    #     # use 6 residual blocks for 128x128 images
    #     res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=6)      # (?, w/4, h/4, 128)
    #   else:
    #     # 9 blocks for higher resolution
    #     res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9)      # (?, w/4, h/4, 128)

    #   # fractional-strided convolution
    #   u64 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
    #       reuse=self.reuse, name='u64')                                 # (?, w/2, h/2, 64)
    #   u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
    #       reuse=self.reuse, name='u32', output_size=self.image_size)         # (?, w, h, 32)

    #   # conv layer
    #   # Note: the paper said that ReLU and _norm were used
    #   # but actually tanh was used and no _norm here
    #   output = ops.c7s1_k(u32, 3, norm=None,
    #       activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)

    # print("Before scoping, the reuse is " + str(self.reuse))
    # Pix2Pix implementation
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:

      s = self.output_size
      s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

      # image is (256 x 256 x input_c_dim)
      # e1 = conv2d(image, self.gf_dim, name='g_e1_conv', reuse=self.reuse)
      # print("At this point, self.reuse is " + str(self.reuse))
      # if self.should_reuse:
      #   self.reuse = True
      e1 = custom_ops.conv2d(input, self.gf_dim, name='g_e1_conv', reuse=self.reuse)
      # e1 is (128 x 128 x self.gf_dim)
      e2 = self.g_bn_e2(custom_ops.conv2d(custom_ops.lrelu(e1), self.gf_dim*2, name='g_e2_conv', reuse=self.reuse))
      # e2 is (64 x 64 x self.gf_dim*2)
      e3 = self.g_bn_e3(custom_ops.conv2d(custom_ops.lrelu(e2), self.gf_dim*4, name='g_e3_conv', reuse=self.reuse))
      # e3 is (32 x 32 x self.gf_dim*4)
      e4 = self.g_bn_e4(custom_ops.conv2d(custom_ops.lrelu(e3), self.gf_dim*8, name='g_e4_conv', reuse=self.reuse))
      # e4 is (16 x 16 x self.gf_dim*8)
      e5 = self.g_bn_e5(custom_ops.conv2d(custom_ops.lrelu(e4), self.gf_dim*8, name='g_e5_conv', reuse=self.reuse))
      # e5 is (8 x 8 x self.gf_dim*8)
      e6 = self.g_bn_e6(custom_ops.conv2d(custom_ops.lrelu(e5), self.gf_dim*8, name='g_e6_conv', reuse=self.reuse))
      # e6 is (4 x 4 x self.gf_dim*8)
      e7 = self.g_bn_e7(custom_ops.conv2d(custom_ops.lrelu(e6), self.gf_dim*8, name='g_e7_conv', reuse=self.reuse))
      # e7 is (2 x 2 x self.gf_dim*8)
      e8 = self.g_bn_e8(custom_ops.conv2d(custom_ops.lrelu(e7), self.gf_dim*8, name='g_e8_conv', reuse=self.reuse))
      # e8 is (1 x 1 x self.gf_dim*8)

      self.d1, self.d1_w, self.d1_b = custom_ops.deconv2d(tf.nn.relu(e8),
          [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True, reuse=self.reuse)
      d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
      d1 = tf.concat([d1, e7], 3)
      # d1 is (2 x 2 x self.gf_dim*8*2)

      self.d2, self.d2_w, self.d2_b = custom_ops.deconv2d(tf.nn.relu(d1),
          [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True, reuse=self.reuse)
      d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
      d2 = tf.concat([d2, e6], 3)
      # d2 is (4 x 4 x self.gf_dim*8*2)

      self.d3, self.d3_w, self.d3_b = custom_ops.deconv2d(tf.nn.relu(d2),
          [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True, reuse=self.reuse)
      d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
      d3 = tf.concat([d3, e5], 3)
      # d3 is (8 x 8 x self.gf_dim*8*2)

      self.d4, self.d4_w, self.d4_b = custom_ops.deconv2d(tf.nn.relu(d3),
          [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True, reuse=self.reuse)
      d4 = self.g_bn_d4(self.d4)
      d4 = tf.concat([d4, e4], 3)
      # d4 is (16 x 16 x self.gf_dim*8*2)

      self.d5, self.d5_w, self.d5_b = custom_ops.deconv2d(tf.nn.relu(d4),
          [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True, reuse=self.reuse)
      d5 = self.g_bn_d5(self.d5)
      d5 = tf.concat([d5, e3], 3)
      # d5 is (32 x 32 x self.gf_dim*4*2)

      self.d6, self.d6_w, self.d6_b = custom_ops.deconv2d(tf.nn.relu(d5),
          [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True, reuse=self.reuse)
      d6 = self.g_bn_d6(self.d6)
      d6 = tf.concat([d6, e2], 3)
      # d6 is (64 x 64 x self.gf_dim*2*2)

      self.d7, self.d7_w, self.d7_b = custom_ops.deconv2d(tf.nn.relu(d6),
          [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True, reuse=self.reuse)
      d7 = self.g_bn_d7(self.d7)
      d7 = tf.concat([d7, e1], 3)
      # d7 is (128 x 128 x self.gf_dim*1*2)

      self.d8, self.d8_w, self.d8_b = custom_ops.deconv2d(tf.nn.relu(d7),
          [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True, reuse=self.reuse)
      # d8 is (256 x 256 x output_c_dim)

      # return tf.nn.tanh(self.d8)
      output = tf.nn.tanh(self.d8)

    # set reuse=True for next call
    self.reuse = True
    # self.should_reuse = True
    # print("Just set reuse variable to " + str(self.reuse))
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    # print("Reuse after the variables call is now: " + str(self.reuse))

    # print("Made it to the end of the first call!")

    # print("The type of the output is: " + str(type(output)))

    return output

  def sample(self, input):
    image = utils.batch_convert2int(self.__call__(input))
    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return image
