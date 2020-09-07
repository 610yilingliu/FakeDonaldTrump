import cv2
import os
import pandas as pd
import numpy as np

def to_npz(path, outname):
    if not os.path.exists(path):
        print(path + ' not exists')
        return
    x = []
    files = os.listdir(path)
    for f in files:
        img = cv2.imread(path + '/' + f)
        x.append(img)
    np.savez(outname, train = x)


if __name__ == '__main__':
    to_npz('./trump', 'trump_np')
    # f = np.load('trump_np.npz')
    # f.files