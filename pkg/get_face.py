import os
import dlib
import cv2
import face_recognition
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
        if r - l < 60 or d - u < 60:
            return
        # enlarger the selection
        extend_d = int((d - u) * 0.02)
        extend_u = int((d - u) * 0.18)
        extend_lr = int((r - l) * 0.1)
        new_u = u - extend_u
        new_d = d + extend_d
        new_l = l - extend_lr
        new_r = r + extend_lr
        if new_u < 0 or new_d > image.shape[0] or new_l < 0 or new_d > image.shape[1]:
            return
        cropped = image[new_u:new_d, l:r]
        regularized = cv2.resize(cropped, (64, 64))
        yield regularized, idx

def photo_cleaner(photo, valid_photo):
    valid_image = face_recognition.load_image_file(valid_photo)
    valid_encoding = face_recognition.face_encodings(valid_image)[0]
    image = face_recognition.load_image_file(photo)
    image_encoding = face_recognition.face_encodings(image)
    if image_encoding:
        image_encoding = image_encoding[0]
    else:
        os.remove(photo)
        return
    results = face_recognition.compare_faces([valid_encoding], image_encoding)
    if results[0] == False:
        os.remove(photo)

if __name__ == '__main__':
    pics = "./trump_photos"
    for f in os.listdir(pics):
        p = pics + '/' + f
        for pic, idx in cut_image(p):
            path = './trump/' + str(idx) + '_' + f
            cv2.imwrite(path, pic)
            photo_cleaner(path, 'trump.png')