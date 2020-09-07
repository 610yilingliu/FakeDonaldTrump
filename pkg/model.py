import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, models , optimizers , losses
from tensorflow.keras.layers import *

batch_size = 128
epochs = 100
k1 = (3, 3)
k2 = (4, 4)
s1 = (2, 2)



def gan_model(d_model, g_model):
    d_model.trainable = False
    model = Sequential([d_mode, g_model])
    model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr = 0.0002, beta_1 = 0.5))
    return model

def discriminator(input_size):
    # 2 * 2 stride has the same effect with pooling layer
    D = Sequential()
    D.add(Conv2D(filters = 64, kernel_size = k1, padding = 'same', input_shape = input_size))
    D.add(LeakyReLU(alpha = 0.2))

    D.add(Conv2D(filters = 128, kernel_size = k1, padding = 'same', strides = s1))
    D.add(LeakyReLU(alpha = 0.2))
    D.add(Dropout(0.25))

    D.add(Conv2D(filters = 256, kernel_size = k1, padding = 'same', strides = s1))
    D.add(LeakyReLU(alpha = 0.2))
    D.add(Dropout(0.25))

    D.add(Flatten())
    D.add(Dropout(0.25))
    D.add(Dense(1, activation = 'sigmoid'))

    D.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5), metrics = ['accuracy'])
    return D

def generator(latent_dim):
    G = Sequential()
    G.add(Dense(filters = 256, units = base << 4, input_dim = latent_dim))
    G.add(LeakyReLU(alpha = 0.2))
    G.add(Reshape((4,4,256)))
    # to 8 * 8
    G.add(Conv2DTranspose(filters = 128, kernel_size = k2, padding = 'same', strides = (2,2)))
    G.add(LeakyReLU(alpha = 0.2))
    # to 16 * 16
    G.add(Conv2DTranspose(filters = 128, kernel_size = k2, padding = 'same', strides = (2,2)))
    G.add(LeakyReLU(alpha = 0.2))
    # to 32 * 32
    G.add(Conv2DTranspose(filters = 128, kernel_size = k2, padding = 'same', strides = (2,2)))
    G.add(LeakyReLU(alpha = 0.2))
    # to 64 * 64
    G.add(Conv2DTranspose(filters = 128, kernel_size = k2, padding = 'same', strides = (2,2)))
    G.add(LeakyReLU(alpha = 0.2))
    G.add(Conv2D(filters = 3, kernel_size = k1, activation = 'tanh', padding = 'same'))
    return G


def generate_latent_points(latent_dim,n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples,latent_dim)
    return x_input    

def generate_real_samples(dataset, n_samples):
    # define random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    x = dataset[ix]
    # generate class label (label = 1)
    y = np.ones((n_samples,1))
    return x,y

def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    x = g_model.predict(x_input)
    # generate class label (label = 0)
    y = np.zeros((n_samples,1))
    return x,y

def load_data(path):
    """
    :type path: String, .npz file path. 
    ### Remember that this npz file only contains one file, i.e. train. If you have alternative names of the dataset, please use data.files to view keys
    :rtype: numpy dataset
    """
    data = np.load(path)
    return data['train']
    
