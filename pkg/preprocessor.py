import os
import dlib
import cv2
import numpy as np

def cut_image(path):
    """
    :type path: string, path of a specific image
    :ytype cutted image (numpy)
    """
    if not os.path.exists(path):
        print(path + ' not exists')
        return
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    dets = detector(image, 1)
    if not dets:
        return
    for idx, face in enumerate(dets):
        l = face.left()
        r = face.right()
        u = face.top()
        d = face.bottom()
        # enlarger the selection
        extend_d = int((d - u) * 0.05)
        extend_u = int((d - u) * 0.15)
        new_u = u + extend_u
        new_d = d + extend_d
        cropped = image[new_u:new_d, l:r]
        regularized = cv2.resize(cropped, (78, 64))
        yield regularized


