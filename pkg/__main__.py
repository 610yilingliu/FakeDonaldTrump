## libraries like numpy, tensorflow, cv2 are all imported by *
from get_npz import *
from dcgan import *
from preprocess import *
from visualize_tools import *

# Learning is a static value or not
dynamic_lr = False


logname = time_helper()
if not os.path.exists('./logs'):
    os.mkdir('logs')
## put things shown in console in a .log file
sys.stdout = Logger('./logs/' + logname + '.log')



SOURCE_PATH = './trump'
NP_NAME = 'trump_np.npz'
FILE_PATH = './data'
## save the npz file for feature use, you can also store it in memory without export and re-read it. I/O spend a lots of time
## better explain it in coding interview

if not os.path.exists(FILE_PATH + '/' + NP_NAME):
    pic_to_npz(SOURCE_PATH, NP_NAME, target_folder = FILE_PATH)

train_images = None
for np_arr in import_npz(FILE_PATH + '/' + NP_NAME):
    train_images = np_arr
    ## we just need that x_train. You can also use return instead of yield in import_npz, but yield is much general for .npz datasets.
    ## that function belongs to the wheels that I pre-defined to use in different projects.
    break

show_imgs(train_images, './report_imgs', specific_suf = 'source_pics', show_num = 20)

total_num = train_images.shape[0]
print('There are ' + str(total_num) + ' images in the current dataset')

train_images = regularize(train_images)
train_images = scale_img(train_images)

BUFFER_SIZE = 60000
batch_size = 512

batch_num = total_num // batch_size + 1
## Batch and shuffle the data. Reference: https://www.tensorflow.org/tutorials/generative/dcgan
## Similar with Yuliya's code in cell 3

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(batch_size).repeat()
train_data_iter = iter(train_dataset)


conv_dim = 64
z_size = 100

D = Discriminator(conv_dim)
D.build(input_shape=(batch_size, 3, 64, 64))
print(D.summary())

G = Generator(z_size = z_size, conv_dim= conv_dim)
G.build(input_shape =(batch_size, z_size))
print(G.summary())

# params suggested to be good in the DCGAN paper 
# GANs can be very seinsitive to small changes in parameter values
# Check out some options in the DCGAN paper:
#    https://arxiv.org/pdf/1511.06434.pdf

if dynamic_lr == False:
    lr = 0.0002
else:
    lr = 0.02

beta1=0.5
beta2=0.999 # default value OF DCGAN


d_optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = beta1, beta_2 = beta2)
g_optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = beta1, beta_2 = beta2)

# training hyperparameters
num_epochs = 400 # Yuliya's notebook says that 50 epochs will run for a 2-4 hours - maybe it is a data from mac cpu?


# keep track of loss
losses = []

## export result every n epochs
output_freq = 10

# Get some fixed data for evaluating the model's performance as we train
# Keep these images constant
## it is just for showing result! if you do not want to visualize the result images it could be ignored
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = tf.convert_to_tensor(fixed_z, dtype=tf.float32)

## save model
## Reference: https://www.tensorflow.org/tutorials/generative/dcgan
checkpoint_dir = './training_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=g_optimizer,
                                 discriminator_optimizer=d_optimizer,
                                 generator=G,
                                 discriminator=D)

# change learning rate while training
def change_lr(new_lr):
    d_optimizer.lr.assign(new_lr)
    g_optimizer.lr.assign(new_lr)

losses = []

## Defined that one epoch is to go through the whole dataset for once
whole_start = time.time()
for epoch in range(num_epochs):
    start = time.time()

    if dynamic_lr:
        if epoch == 50:
            lr = 0.01
            change_lr(lr)
        elif epoch == 100:
            lr = 0.005
            change_lr(lr)
        elif epoch == 150:
            lr = 0.002
        elif epoch == 200:
            lr = 0.001
        elif epoch == 300:
            lr = 0.0002
    ## In tensorflow the gradient will be reset to zero for each training automatically
    for batch_i, real_images in enumerate(train_data_iter):
        ## custom break to enter another epoch, otherwise the iterator will be repeat forever
        if batch_i == batch_num:
            break
        zz = np.random.uniform(-1, 1, size=(batch_size, z_size))
        noise = tf.convert_to_tensor(zz, dtype=tf.float32)

        with tf.GradientTape() as tape:
            d_loss = dis_loss(G, D, noise, real_images, is_training = True)
        grads = tape.gradient(d_loss, D.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, D.trainable_variables))


        with tf.GradientTape() as tape:
            g_loss = gen_loss(G, D, noise, is_training = True)
        grads = tape.gradient(g_loss, G.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, G.trainable_variables))
        ## give an output for every 20 batch for you to ensure that your code is working
        if batch_i % 20 == 0 or batch_i == batch_num - 1:
            print(str(batch_i + 1) + '/' + str(batch_num) + ' batch in epoch ' + str(epoch + 1) + ' is training')
    
    # show picture result and save checkpoint for every output_freq epochs
    if (epoch + 1) % output_freq == 0 or epoch == num_epochs - 1 or epoch == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
        out = G(fixed_z, training = False)
        out = out.numpy()
        out = de_project(out)
        show_imgs(out, './report_imgs', specific_suf = 'epoch' + str(epoch + 1), show_num = 16)
    # every loss will be visualize
    losses.append((float(d_loss), float(g_loss)))
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    print('Loss for current epoch: ' + 'D ' + str(losses[-1][0]) + '| G ' + str(losses[-1][1]))

total_time = time.time() - whole_start
print('Training finished, spend ' + str(total_time//60) + ' minutes ' + str(total_time % 60) + 'secs')

loss_graph(losses, './report_imgs', 'losses.png')