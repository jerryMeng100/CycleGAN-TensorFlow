import tensorflow as tf
import ops

class Discriminator:
  def __init__(self, name, is_training, norm='instance', use_sigmoid=False):
    self.name = name
    self.is_training = is_training
    self.norm = norm
    self.reuse = False
    self.use_sigmoid = use_sigmoid

  def __call__(self, input):
    """
    Args:
      input: batch_size x image_size x image_size x 3
    Returns:
      output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
              filled with 0.9 if real, 0.0 if fake
    """
    with tf.variable_scope(self.name):
      # convolution layers
      C64 = ops.Ck(input, 64, reuse=self.reuse, norm=None,
          is_training=self.is_training, name='C64')             # (?, w/2, h/2, 64)
      C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C128')            # (?, w/4, h/4, 128)
      C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C256')            # (?, w/8, h/8, 256)
      C512 = ops.Ck(C256, 512,reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C512')            # (?, w/16, h/16, 512)

      # apply a convolution to produce a 1 dimensional output (1 channel?)
      # use_sigmoid = False if use_lsgan = True
      output = ops.last_conv(C512, reuse=self.reuse,
          use_sigmoid=self.use_sigmoid, name='output')          # (?, w/16, h/16, 1)

    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output

class Discriminator16x16:
  def __init__(self, name, is_training, norm='instance', use_sigmoid=False):
    self.name = name
    self.is_training = is_training
    self.norm = norm
    self.reuse = False
    self.use_sigmoid = use_sigmoid

  def __call__(self, input):
    """
    Args:
      input: batch_size x image_size x image_size x 3
    Returns:
      output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
              filled with 0.9 if real, 0.0 if fake
    """
    with tf.variable_scope(self.name):
      # convolution layers
      C64 = ops.Ck(input, 64, reuse=self.reuse, norm=None,
          is_training=self.is_training, name='C64')             # (?, w/2, h/2, 64)
      C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C128')            # (?, w/4, h/4, 128)

      # apply a convolution to produce a 1 dimensional output (1 channel?)
      # use_sigmoid = False if use_lsgan = True
      output = ops.last_conv(C128, reuse=self.reuse,
          use_sigmoid=self.use_sigmoid, name='output')          # (?, w/16, h/16, 1)

    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output

# class PixelDiscriminator(nn.Module):
#     """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
#
#     def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
#         """Construct a 1x1 PatchGAN discriminator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             norm_layer      -- normalization layer
#         """
#         super(PixelDiscriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func != nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer != nn.InstanceNorm2d
#
#         self.net = [
#             nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
#             norm_layer(ndf * 2),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
#
#         self.net = nn.Sequential(*self.net)