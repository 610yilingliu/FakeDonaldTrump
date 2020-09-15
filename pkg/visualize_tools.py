## Remember the functions in this file are not exactly follow Yuliya's code
## Most of the code here are the wheels I made before for general use
import os
import time
import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from get_npz import make_dir

def de_project(np_arr):
    """
    project numpy array from [-1, 1] to [0, 255] for visualization
    """
    item = (np_arr +1)*255 / 2
    return item.astype(np.int32, copy=True) 

def time_helper(seperator = '_', to_sec = False):
    """
    return a string like 2020_09_11_22_43_00 (if to_sec is True) or 2020_09_11_22_43 (if to_sec is False)
    """
    localtime = time.asctime(time.localtime(time.time()))
    if to_sec:
        return time.strftime("%Y" + seperator + "%m" + seperator + "%d" + seperator + "%H" + seperator + "%M" + seperator + "%S", time.localtime()) 
    return time.strftime("%Y" + seperator + "%m" + seperator + "%d" + seperator + "%H" + seperator + "%M", time.localtime()) 

def show_imgs(np_arr, dest_path, specific_sep = None, specific_suf = None, show_num = None, show = False):
    """
    :type np_arr: numpy array contains images
    :type dest_path: String, destination folder to export images
    :type specific_sep: String, seperator inside output file name. like 2020_01_01_01_01.png or 202001010101.png
    :type specific_suf: String, if you specific a suffix like a, the filename will be 2020_01_02_01_01_a.png
    :type show_num: int. How many number of pictures will be shown in the plot
    """
    make_dir(dest_path)
    if show_num is None:
        data_size = np_arr.shape[0]
    else:
        data_size = min(show_num, np_arr.shape[0])
    plot_size = int(math.sqrt(data_size))
    l = plot_size
    if l == 0:
        print('empty numpy array')
        return
    h = data_size // l if data_size % l == 0 else (data_size // l) + 1
    plt.figure(figsize=(8, 8))
    for i in range(data_size):
        p = plt.subplot(h, l, i + 1)
        p.axis('off')
        # HWC to CHW
        new = np_arr[i].transpose((1, 2, 0))
        plt.imshow(new)
    f = plt.gcf()
    if show:
        plt.show()
        
    plt.draw()
    if specific_suf and specific_sep:
        fname = time_helper(specific_sep) + specific_sep + specific_suf
    elif specific_sep:
        fname = time_helper(specific_sep)
    elif specific_suf:
        fname = time_helper() + '_' + specific_suf
    else:
        fname = time_helper()
    name = dest_path + '/' + fname + '.png'
    f.savefig(name)
    try:
        f.savefig(name)
    except:
        print('The file name you defined is ' + name + ', please make sure that there are no system-rejected symbol in file name.\n')
        print('Program terminated...')
        exit(0)
    plt.close()

## Export system output to a log file
class Logger(object):
    def __init__(self, filename='dcgan.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass


def loss_graph(losses, dest_path, name):
    make_dir(dest_path)
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.plot(losses.T[1], label='Generator', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    f = plt.gcf()
    plt.draw()
    f.savefig(dest_path + '/' + name)
    plt.close()
