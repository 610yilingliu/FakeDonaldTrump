import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


def deconv(out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Create a transposed-convolutional layer with optional batch normalization """
    # create a sequence of transpose + optional batch norm layers

    ## We don't need that in_channel in tensorflow
    layers = []
    transpose_conv_layer = Conv2DTranspose(out_channels, kernel_size, strides = stride, padding = 'same', use_bias = False, data_format = "channels_first")
    # append transpose convolutional layer
    layers.append(transpose_conv_layer)
    
    if batch_norm:
        # append batchnorm layer
        layers.append(BatchNormalization())
    
    ## rtype: List[t_conv_layer, batch_norm] or List[t_conv_Layer]
    return layers

class Generator(keras.Model):
    ## outputsize = stride * (inputsize - 1) + 2 * padding - kernelsize + 2. if padding == 1 than outputsize == inputsize. So we use padding = 'same' in tf
    def __init__(self, z_size, conv_dim = 32):
        ## inherit init method from class Model in keras, if you have no idea with what inherit methods from
        ## parent model, please Google "super python"
        super(Generator, self).__init__()
        # complete init function
        
        self.conv_dim = conv_dim
        self.fc = Dense(conv_dim * 4 * 4 * 4, input_shape = (z_size,))

        t_conv0 = deconv(conv_dim * 8, 4)
        self.t_conv0 = t_conv0[0]
        if len(t_conv0) == 2:
            self.bn_0 = t_conv0[1]

        t_conv1 = deconv(conv_dim * 4, 4)
        self.t_conv1 = t_conv1[0]
        if len(t_conv1) == 2:
            self.bn_1 = t_conv1[1]

        t_conv2 = deconv(conv_dim * 2, 4)
        self.t_conv2 = t_conv2[0]
        if len(t_conv2) == 2:
            self.bn_2 = t_conv2[1]

        # desired depth for RGB image is 3
        ## output here is in CHW format
        self.t_conv3 = deconv(3, 4, batch_norm = False)[0]
    
    def call(self, xx, training = None):
        # call in tf is an equivalent with forward in torch
        out = self.fc(xx)
        out = tf.reshape(out, [-1, self.conv_dim * 4, 4, 4])
        out = self.t_conv0(out)
        if self.bn_0:
            out = self.bn_0(out, training = training)  
        out = tf.nn.relu(out)

        out = self.t_conv1(out)
        if self.bn_1:
            out = self.bn_1(out, training = training)
        out = tf.nn.relu(out)

        out = self.t_conv2(out)
        if self.bn_2:
            out = self.bn_2(out, training = training)
        out = tf.nn.relu(out)

        out = self.t_conv3(out)
        out = tf.tanh(out)
        ## to HWC format
        ## Time complexity of numpy.transpose is O(1), according to: https://www.thetopsites.net/article/58279082.shtml
        # out = tf.transpose(out, perm = [0, 3, 1, 2])
        return out

def conv(out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = Conv2D(out_channels, kernel_size, strides = stride, padding = 'same', use_bias = False, data_format = "channels_first")
    # bias is set to False, so the layers are not offset by any amount
    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(BatchNormalization())
     
    ## rtype: List[conv_layer, batch_norm] or List[conv_layer]
    return layers
    
class Discriminator(keras.Model):
    ## outputsize = (inputsize - kernelsize + 2 * padding)/stride + 1, so when stride = 2, kernel_size = 4. if padding == 1 than outputsize == inputsize. So we use padding = 'same' in tf
    ## if you want to custom padding size, please read helper here https://stackoverflow.com/questions/37659538/custom-padding-for-convolutions-in-tensorflow
    ## tf.pad is still available in tf 2.0+
    ## you can also create a sequence and use sequence.add(layer) to add layers to model, see the tutorial here: 
    ## https://www.tensorflow.org/tutorials/generative/dcgan
    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(conv_dim, 4, batch_norm= False)[0]

        conv2 = conv(conv_dim * 2, 4)
        self.conv2 = conv2[0]
        if len(conv2) == 2:
            self.bn_1 = conv2[1]

        conv3 = conv(conv_dim * 4, 4)
        self.conv3 = conv3[0]
        if len(conv3) == 2:
            self.bn_2 = conv3[1]
        
        conv4 = conv(conv_dim * 8, 4)
        self.conv4 = conv4[0]
        if len(conv4) == 2:
            self.bn_3 = conv4[1]

        self.flatten = Flatten()
        self.fc = Dense(1)
        
    def call(self, xx, training = None):
        out = self.conv1(xx)
        out = tf.nn.leaky_relu(out, alpha = 0.2)

        out = self.conv2(out)
        if self.bn_1:
            out = self.bn_1(out, training = training)
        out = tf.nn.leaky_relu(out, alpha = 0.2)

        out = self.conv3(out)
        if self.bn_2:
            out = self.bn_2(out, training = training)

        out = self.conv4(out)
        if self.bn_3:
            out = self.bn_3(out, training = training)
        out = self.flatten(out) 

        out = self.fc(out)
        return out

def real_loss(D_out, smooth=False):
    batch_size = D_out.shape[0]
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = tf.ones(batch_size) * 0.9
    else:
        labels = tf.ones(batch_size) # real labels = 1
    ## Reference 1: https://stackoverflow.com/questions/55683729/bcewithlogitsloss-in-keras
    ## Reference 2: https://www.tensorflow.org/tutorials/generative/dcgan
    ## So we use BinaryCrossentropy here in tf to replace BCEWithLogitsLoss() in torch
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss = criterion(labels, D_out)
    return loss

def fake_loss(D_out):
    batch_size = D_out.shape[0]
    labels = tf.zeros(batch_size) # fake labels = 0
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # calculate loss
    loss = criterion(labels, D_out)
    return loss

## I put in the loss calculation here instead of main function
def dis_loss(generator, discriminator, input_noise, real_image, is_training):
    fake_image = generator(input_noise, is_training)
    d_real_logits = discriminator(real_image, is_training)
    d_fake_logits = discriminator(fake_image, is_training)

    d_loss_real = real_loss(d_real_logits)
    d_loss_fake = fake_loss(d_fake_logits)
    loss = d_loss_real + d_loss_fake
    return loss

def gen_loss(generator, discriminator, input_noise, is_training):
    fake_image = generator(input_noise, is_training)
    fake_loss = discriminator(fake_image, is_training)
    loss = real_loss(fake_loss)
    return loss