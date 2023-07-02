import numpy as np
import cv2
from PIL import Image

im_ori = Image.open('./data/image-parse-v3/person.png')
im = Image.open('./data/parse/person.png')
print(np.unique(np.array(im_ori)))
print(np.unique(np.array(im)))

np_im = np.array(im)
np_im[np_im==2] = 151
np_im[np_im==9] = 178
np_im[np_im==13] = 191
np_im[np_im==14] = 221
np_im[np_im==15] = 246
cv2.imwrite("data/test.png", np_im)
# Image.fromarray(np_im)