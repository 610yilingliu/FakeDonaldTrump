import tensorflow as tf
import numpy as np


def import_npz(path):
    """
    If you have no sense about how .npz file work, please explore it by the following code and IDE(with break point)
    ```
    import numpy as np
    data = np.load(your_npz_file_path)
    data.files
    ```
    And you will see a list contains dictionary keys. For example, ['a','b']. If you type d = data['a'] and d.shape, you will see the numpy shape. 
    That d is numpy array
    :type path: String
    :ytype: numpy array
    """
    f = np.load(path)
    files = f.files
    for dataset in files:
        yield f[dataset]


# The output of our G will be tanh activated e.i. on a range from [-1 1], 
# We need to rescale training images to that range. (Currently: [0 1])
def scale_img(np_arr, feature_range=(-1, 1)):
    """
    :type np_arr: Numpy Array of images.
    Could be one image or numpy array contains multiple images, it is not important because division method between numpy array
    and number doesn't care about the shape of numpy data
    :type feature_range: Tuple, project numbers from [0, 1] to [feature_range[0], feature_range[1]]
    :rtype: Projected Numpy Array
    """
    # assume np_arr is scaled to (0, 1)
    # scale to feature_range and return scaled np_arr
    ## min and max are not good variable name if you are using IDE because there is a function min(list) and a function max(list)
    ## better not be the same with these functions, especially in the coding part of an interview
    mi, mx = feature_range
    np_arr = np_arr * (mx - mi) + mi
    return np_arr

def regularize(np_arr):
    """
    Project from [0, 255] to [0, 1]
    :type np_arr: numpy array
    :rtype: Projected numpy array
    """
    return np_arr/255


    

        

