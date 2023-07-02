#title Upload images from your computer
#markdown Description of parameters
#markdown - `SHOW_FULLSIZE`  - Shows image in full size (may take a long time to load)
#markdown - `PREPROCESSING_METHOD`  - Preprocessing method
#markdown - `SEGMENTATION_NETWORK`  - Segmentation network. Use `u2net` for hairs-like objects and `tracer_b7` for objects
#markdown - `POSTPROCESSING_METHOD`  - Postprocessing method
#markdown - `SEGMENTATION_MASK_SIZE` - Segmentation mask size. Use 640 for Tracer B7 and 320 for U2Net
#markdown - `TRIMAP_DILATION`  - The size of the offset radius from the object mask in pixels when forming an unknown area
#markdown - `TRIMAP_EROSION`  - The number of iterations of erosion that the object's mask will be subjected to before forming an unknown area

import os
import numpy as np
from PIL import Image, ImageOps
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface
import cv2

SHOW_FULLSIZE = False #param {type:"boolean"}
PREPROCESSING_METHOD = "none" #param ["stub", "none"]
SEGMENTATION_NETWORK = "tracer_b7" #param ["u2net", "deeplabv3", "basnet", "tracer_b7"]
POSTPROCESSING_METHOD = "fba" #param ["fba", "none"] 
SEGMENTATION_MASK_SIZE = 640 #param ["640", "320"] {type:"raw", allow-input: true}
TRIMAP_DILATION = 30 #param {type:"integer"}
TRIMAP_EROSION = 5 #param {type:"integer"}
DEVICE = 'cuda' # 'cuda'

config = MLConfig(segmentation_network=SEGMENTATION_NETWORK,
                  preprocessing_method=PREPROCESSING_METHOD,
                  postprocessing_method=POSTPROCESSING_METHOD,
                  seg_mask_size=SEGMENTATION_MASK_SIZE,
                  trimap_dilation=TRIMAP_DILATION,
                  trimap_erosion=TRIMAP_EROSION,
                  device=DEVICE)

interface = init_interface(config)

imgs = []
root = '/root/HR-VITON/data/test/cloth'
for name in os.listdir(root):
    imgs.append(root + '/' + name)

os.makedirs("/root/HR-VITON/data/test/cloth-mask", exist_ok =  True)
images = interface(imgs)
for i, im in enumerate(images):
    img = np.array(im)
    img = img[...,:3] # no transparency
    idx = (img[...,0]==130)&(img[...,1]==130)&(img[...,2]==130) # background 0 or 130, just try it
    img = np.ones(idx.shape)*255
    img[idx] = 0
    im = Image.fromarray(np.uint8(img), 'L')
    im.save(f'/root/HR-VITON/data/test/cloth-mask/{imgs[i].split("/")[-1].split(".")[0]}.jpg')