from pkg.preprocessor import *

pics = "./trump_photos"
for f in os.listdir(pics):
    p = pics + '/' + f
    for pic in cut_image(p):
        cv2.imwrite('./trump/' + f, pic)