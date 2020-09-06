import tensorflow as tf
from tensorflow.keras import Sequential, models , optimizers , losses
from tensorflow.keras.layers import *

batch_size = 128
epochs = 100
k1 = (3, 3)
k2 = (4, 4)
s1 = (2, 2)

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
