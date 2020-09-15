import cv2
import os
import scipy.io
import numpy as np


def pic_to_npz(path, outname, target_folder = None):
    """
    :type path: String, folder that stores pictures
    :type outname: String, name of the output .npz file. Remember you do not have to input '.npz' suffix.
    :type target_folder: root path of the 
    """
    if not os.path.exists(path):
        print(path + ' not exists')
        return
    x = []
    files = os.listdir(path)
    for f in files:
        img = cv2.imread(path + '/' + f)
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.transpose((2,0,1))
        x.append(img)
    if target_folder is None:
        np.savez(outname, train = x)
    else:
        make_dir(target_folder)
        np.savez(target_folder + '/' + outname, train = x)

def mat_tonpz(mat_file, outname, target_folder = None):
    """
    :type mat_file: String, path of a .mat file
    :type outname: String, name of the output(not the whole path name)
    :type target_folder: String, name of the folder to export .npz file
    
    ### Explain:

    After you read `.mat` file with scipy, the data that stores in your memory is a dictionary. For dataset that Yuliya is using,
    the dictionary has two keys with data: 'X' and 'y' (You can view the datatype by add some break points in your ide(not Jupyter Notebook!)), and enter debug mode to run. The data we export is numpy in 'X', which are images we want.
    One thing you have to be careful in .mat file is that the first three keys in that file are not data, these keys are listed in set invalid in the 
    following code, you need to skip it
    """
    data = scipy.io.loadmat(mat_file)
    invalid = ('__header__', '__version__', '__globals__')
    for key in data:
        if key in invalid:
            continue
        ## Only store X data
        train_data = data[key]
        ## Original data is in HWCN style (height, width, channel, images). TF GPU version calculates NCHW faster, but matplotlib.pyplot and TF CPU version requires NHWC. 
        ## Transpose it into NCHW for tf, and we will do NHWC for plotting in visualize_tools.show_image()
        ## The reason why GPU prefers NCHW format explained here(in Chinese, use Google translate if you cannot read it):
        ## https://blog.csdn.net/weixin_37801695/article/details/86614566
        train_data = train_data.transpose((3, 2, 0, 1))
        if target_folder is None:
            np.savez(outname, train = train_data)
        else: 
            make_dir(target_folder)
            np.savez(target_folder + '/' + outname, train = train_data)
        break


def make_dir(path):
    """
    :type path: String. Only separator '/' is supported
    
    This is a helper function to build an directory. os.mkdir does not allow you to create floder like `./abc/def`, you must create `./abc` first
    and create `./abc/def` inside it
    """
    path = path.split('/')
    cur = './'
    for i in range(len(path)):
        if path[i] == '' or path[i] == '.':
            continue
        cur = cur + '/' + path[i]
        if not os.path.exists(cur):
            try:
                os.mkdir(cur)
            except:
                print('Something Strange happend while creating directory ' + cur + '\n')
                return