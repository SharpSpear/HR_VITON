# import os
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from cloth_segmentation.data.base_dataset import Normalize_image
# from cloth_segmentation.utils.saving_utils import load_checkpoint_mgpu
# from cloth_segmentation.networks.u2net import U2NET
# import base64

# data_folder = "test"

# cls = [11, 12, 13, 14, 15]

# def human_parsing_atr(input_file):
#     import sys
#     sys.path.append("Self-Correction-Human-Parsing")
#     os.makedirs("tmp", exist_ok = True)
#     os.system(f"cp {input_file} tmp")
#     cmd = f"python Self-Correction-Human-Parsing/simple_extractor.py --dataset atr \
#             --model-restore Self-Correction-Human-Parsing/checkpoints/atr.pth --input-dir tmp --output-dir {data_folder}"
#     os.system(cmd)
#     os.system(f"rm -rf tmp")

# human_parsing("/root/workspace/source/HR-VITON/database/image/Man_Pink_1.jpg")

# import cv2
# import numpy as np
# from PIL import Image


# org_img1 = cv2.imread("/root/workspace/source/HR-VITON/database/image/Man_Pink_1.jpg")

# org_img = cv2.imread("/root/workspace/source/HR-VITON/Self-Correction-Human-Parsing/test_images/image.png")
# mask_img = np.zeros_like(org_img)
# im_parse = Image.open("/root/workspace/source/HR-VITON/test/Man_Pink_1.png")

# label_array = np.array(im_parse)
# parse_skin = ((label_array == 11).astype(np.float32) +
#             (label_array == 12).astype(np.float32) +
#             (label_array == 13).astype(np.float32) +
#             (label_array == 14).astype(np.float32) +
#             (label_array == 15).astype(np.float32))

# org_img[parse_skin == True] = org_img1[parse_skin == True]

# cv2.imwrite("test.png", org_img)

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface

SHOW_FULLSIZE = False #param {type:"boolean"}
PREPROCESSING_METHOD = "none" #param ["stub", "none"]
SEGMENTATION_NETWORK = "tracer_b7" #param ["u2net", "deeplabv3", "basnet", "tracer_b7"]
POSTPROCESSING_METHOD = "fba" #param ["fba", "none"] 
SEGMENTATION_MASK_SIZE = 640 #param ["640", "320"] {type:"raw", allow-input: true}
TRIMAP_DILATION = 30 #param {type:"integer"}
TRIMAP_EROSION = 5 #param {type:"integer"}
DEVICE = 'cpu' # 'cuda'

config = MLConfig(segmentation_network=SEGMENTATION_NETWORK,
                  preprocessing_method=PREPROCESSING_METHOD,
                  postprocessing_method=POSTPROCESSING_METHOD,
                  seg_mask_size=SEGMENTATION_MASK_SIZE,
                  trimap_dilation=TRIMAP_DILATION,
                  trimap_erosion=TRIMAP_EROSION,
                  device=DEVICE)

interface = init_interface(config)

imgs = ["test_image/1.jpg"]
images = interface(imgs)

for i, im in enumerate(images):
    img = np.array(im)
    RGB = img[...,:3] # no transparency
    BGR = RGB[...,::-1]

    idx = (RGB[...,0]==130)&(RGB[...,1]==130)&(RGB[...,2]==130) # background 0 or 130, just try it
    gray = np.ones(idx.shape)*255
    gray[idx] = 0
    
    BGR[idx==True] = (255, 255, 255)
    cv2.imwrite("test.png", BGR)
    
