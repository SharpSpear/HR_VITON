import glob
import cv2
import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
import numpy as np
import sys
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import pathlib
import shutil 

from cloth_segmentation.data.base_dataset import Normalize_image
from cloth_segmentation.utils.saving_utils import load_checkpoint_mgpu
from cloth_segmentation.networks.u2net import U2NET

# model
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

checkpoint_path = "cloth_segmentation/trained_checkpoint/cloth_segm_u2net_latest.pth"
transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(DEVICE)
net = net.eval()

# End model


# def get_img_agnostic(img, parse, pose_data):
#     parse_array = np.array(parse)
#     # parse_head = ((parse_array == 4).astype(np.float32) +
#     #                 (parse_array == 13).astype(np.float32))
#     parse_lower = ((parse_array == 9).astype(np.float32) +
#                     (parse_array == 12).astype(np.float32) +
#                     (parse_array == 16).astype(np.float32) +
#                     (parse_array == 17).astype(np.float32) +
#                     (parse_array == 18).astype(np.float32) +
#                     (parse_array == 19).astype(np.float32))
#     # parse_lower = ((parse_array == 9).astype(np.float32))

    
#     agnostic = img.copy()
#     agnostic_draw = ImageDraw.Draw(agnostic)

#     length_a = np.linalg.norm(pose_data[5] - pose_data[2])
#     length_b = np.linalg.norm(pose_data[12] - pose_data[9])
#     point = (pose_data[9] + pose_data[12]) / 2
#     pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
#     pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
#     r = int(length_a / 16) + 1
    
#     # # mask arms
#     # agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
#     # for i in [2, 5]:
#     #     pointx, pointy = pose_data[i]
#     #     agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
#     # for i in [3, 4, 6, 7]:
#     #     if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
#     #         continue
#     #     agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
#     #     pointx, pointy = pose_data[i]
#     #     agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

#     # mask torso
#     for i in [9, 12]:
#         pointx, pointy = pose_data[i]
#         agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
#     agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
#     agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
#     agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
#     agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

#     # mask neck
#     # pointx, pointy = pose_data[1]
#     # agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
#     # agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
#     agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

#     return agnostic
def get_img_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 7).astype(np.float32) +
                    (parse_array == 5).astype(np.float32) +
                    (parse_array == 14).astype(np.float32) +
                    (parse_array == 15).astype(np.float32))

    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[9] - pose_data[8])
    r = int(length_a / 4) + 1
    for i in [10, 11, 13, 14]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
    for i in [10, 13]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
    pointx, pointy = pose_data[8]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
 
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic



def get_im_parse_agnostic(im_parse, pose_data, w=768, h=1024):
    label_array = np.array(im_parse)
    parse_lower = ((label_array == 9).astype(np.float32))
    # parse_neck = (label_array == 10).astype(np.float32)

    r = 10
    agnostic = im_parse.copy()

    # # mask arms
    # for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
    #     mask_arm = Image.new('L', (w, h), 'black')
    #     mask_arm_draw = ImageDraw.Draw(mask_arm)
    #     i_prev = pose_ids[0]
    #     for i in pose_ids[1:]:
    #         if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
    #             continue
    #         mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
    #         pointx, pointy = pose_data[i]
    #         radius = r*4 if i == pose_ids[-1] else r*15
    #         mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
    #         i_prev = i
    #     parse_arm = (np.array(mask_arm) / 255) * (label_array == parse_id).astype(np.float32)
    #     agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    # agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    return agnostic

def resize_image(person_folder, output_folder):
    print('~~~~~~start the resize_image function')
    dst_h = 1024
    dst_w = 768

    os.makedirs(f"{output_folder}/image", exist_ok = True)
    # os.makedirs(f"{output_folder}/cloth", exist_ok = True)
    # os.makedirs(f"{output_folder}/cloth-mask", exist_ok = True)

    list_imgs = glob.glob(f"{person_folder}/*")
    for path in list_imgs:
        img = cv2.imread(path)
        h, w, _ = img.shape[:]
        ratio_h = dst_h / h 
        ratio_w = dst_w / w 
        ratio = min(ratio_h, ratio_w)
        img = cv2.resize(img, fx = ratio, fy = ratio, dsize = None)
        h, w, _ = img.shape[:]
        top, bottom, left, right = (dst_h - h) // 2, (dst_h - h) // 2 + (dst_h - h) % 2, (dst_w - w) // 2, (dst_w - w) // 2 + (dst_w - w) % 2
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)

        img_name = os.path.splitext(os.path.basename(path))[0]

        cv2.imwrite(f"{output_folder}/image/{img_name}.jpg", img)
        # cv2.imwrite(f"{output_folder}/cloth/{img_name}.jpg", img)
        # cv2.imwrite(f"{output_folder}/cloth-mask/{img_name}.jpg", img)

    print('~~~~~~close the resize_image function')
def openpose(input_folder, output_folder):
    print('~~~~~~start the openpose function')
    os.makedirs(output_folder, exist_ok = True)
    # cmd = f"./openpose/openpose.bin --image_dir {input_folder}/image --hand --disable_blending --display 0 --write_json \
    #         {output_folder}/openpose_json --write_images {output_folder}/openpose_img --model_folder openpose/models"

    # cmd = rf"openpose-master\artifacts\bin\OpenPoseDemo.exe --image_dir tmp\test\image --hand --disable_blending --display 0 --write_json tmp\test\openpose_json --write_images tmp\test\openpose_img --model_folder openpose-master\artifacts\models"
    # cmd = rf"openpose-master/artifacts/bin/OpenPoseDemo.exe --image_dir {input_folder}\image --hand --disable_blending --display 0 --write_json {output_folder}\openpose_json --write_images {output_folder}\openpose_img --model_folder openpose-master\artifacts\models"
    cmd = rf"openpose1/openpose.bin --image_dir {input_folder}/image --hand --disable_blending --display 0 --write_json {output_folder}\openpose_json --write_images {output_folder}\openpose_img --model_folder openpose/models"
   
    os.system(cmd)
    
    print('~~~~~~close the openpose function')
def densepose(input_folder, output_folder):
    print('~~~~~~start the densepose function')
    import sys
    sys.path.append("DensePose")
    cmd = f"python DensePose/apply_net.py show DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
            https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
            {input_folder}/image dp_segm -v --output_dir {output_folder}/image-densepose/"
    os.system(cmd)

    print('~~~~~~close the densepose function')
def human_parsing(input_folder, output_folder):
    print('~~~~~~start the human_parsing function')
    import sys
    sys.path.append("Self-Correction-Human-Parsing")
    cmd = f"python Self-Correction-Human-Parsing/simple_extractor.py \
            --model-restore Self-Correction-Human-Parsing/checkpoints/lip.pth --input-dir {input_folder}/image --output-dir {output_folder}/image-parse-v3"
    os.system(cmd)
    print('~~~~~~close the human_parsing function')

def parse_agnostic(input_folder, output_folder):
    print('~~~~~~start the parse_agnostic function')
    data_path = input_folder
    output_path = os.path.join(output_folder, "image-parse-agnostic-v3.2")
    
    os.makedirs(output_path, exist_ok=True)
    
    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):
        
        # load pose image
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        
        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
        except IndexError:
            print(pose_name)
            continue

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(data_path, 'image-parse-v3', parse_name))

        agnostic = get_im_parse_agnostic(im_parse, pose_data)
        
        agnostic.save(osp.join(output_path, parse_name))

    print('~~~~~~close the parse_agnostic function')
def human_agnostic(input_folder, output_folder):
    print('~~~~~~start the human_agnostic function')
    data_path = input_folder
    output_path = os.path.join(output_folder, "agnostic-v3.2")
    
    os.makedirs(output_path, exist_ok=True)
    
    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):
        
        # load pose image
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        
        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
        except IndexError:
            print(pose_name)
            continue

        # load parsing image
        im = Image.open(osp.join(data_path, 'image', im_name))
        label_name = im_name.replace('.jpg', '.png')
        im_label = Image.open(osp.join(data_path, 'image-parse-v3', label_name))
        agnostic = get_img_agnostic(im, im_label, pose_data)
        agnostic.save(osp.join(output_path, im_name))

    print('~~~~~~close the human_agnostic function')
def crop_cloth(input_folder, output_dir):
    print('~~~~~~start the crop_cloth function')
    output_dir = os.path.join(output_dir, "cloth")
    imgs = []
    root = input_folder
    for name in os.listdir(root):
        if name[-4] != '.':
          continue
        imgs.append(root + '/' + name)

    os.makedirs(output_dir, exist_ok =  True)
    images = interface(imgs)
    for i, (im, path) in enumerate(zip(images, imgs)):
        img = np.array(im)
        img = img[...,:3] # no transparency
        idx = (img[...,0]==130)&(img[...,1]==130)&(img[...,2]==130) # background 0 or 130, just try it
        img = np.ones(idx.shape)*255
        img[idx] = 0

        img = img.astype(np.uint8)
        contours, hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        maxsize = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > maxsize :
                maxsize = cv2.contourArea(cnt)
                best = cnt
                
        x,y,w,h = cv2.boundingRect(best)

        org_img = cv2.imread(path)

        x1 = np.clip(x, 0, org_img.shape[1])
        y1 = np.clip(y, 0, org_img.shape[0])
        x2 = np.clip(x+w, 0, org_img.shape[1])
        y2 = np.clip(y+h, 0, org_img.shape[0])
        
        crop = org_img[y1:y2, x1:x2, :]
        # cv2.drawContours(org_img, contours, -1, (0, 255, 0), 3)
        # org_img = cv2.rectangle(org_img,(new_x,new_y),(new_x+w,new_y+new_h),(0,255,0),2)
        
        cv2.imwrite(f'{output_dir}/{path.split("/")[-1].split(".")[0]}.jpg', crop)

    print('~~~~~~close the crop_cloth function')

def resize_cloth(input_folder, output_folder):
    print('~~~~~~start the resize_cloth function')
    dst_h = 1024
    dst_w = 768

    os.makedirs(f"{output_folder}/cloth", exist_ok = True)

    list_imgs = glob.glob(f"{input_folder}/*")
    for path in list_imgs:
        img = cv2.imread(path)
        h, w, _ = img.shape[:]
        ratio_h = dst_h / h 
        ratio_w = dst_w / w 
        ratio = min(ratio_h, ratio_w)
        img = cv2.resize(img, fx = ratio, fy = ratio, dsize = None)
        h, w, _ = img.shape[:]
        top, bottom, left, right = (dst_h - h) // 2, (dst_h - h) // 2 + (dst_h - h) % 2, (dst_w - w) // 2, (dst_w - w) // 2 + (dst_w - w) % 2
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = 0)

        img_name = os.path.splitext(os.path.basename(path))[0]

        cv2.imwrite(f"{output_folder}/cloth/{img_name}.jpg", img)

    print('~~~~~~close the resize_cloth function')
def rm_cloth_background(input_folder, output_dir):
    print('~~~~~~start the rm_cloth_background function')
    output_dir = os.path.join(output_dir, "cloth-mask")

    imgs = []
    root = input_folder
    for name in os.listdir(root):
        imgs.append(root + '/' + name)

    os.makedirs(output_dir, exist_ok =  True)
    for image_path in imgs:
        img = Image.open(image_path).convert("RGB")
        image_tensor = transform_rgb(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        output_tensor = net(image_tensor.to(DEVICE))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()
        
        mask = np.zeros_like(output_arr)
        mask[output_arr == 255] = 255
        image_name = os.path.basename(image_path)
        cv2.imwrite(f'{output_dir}/{image_path.split("/")[-1].split(".")[0]}.jpg', mask)

    print('~~~~~~close the rm_cloth_background function')
    # images = interface(imgs)
    # for i, im in enumerate(images):
    #     img = np.array(im)
    #     img = img[...,:3] # no transparency
    #     idx = (img[...,0]==130)&(img[...,1]==130)&(img[...,2]==130) # background 0 or 130, just try it
    #     img = np.ones(idx.shape)*255
    #     img[idx] = 0

    #     im = Image.fromarray(np.uint8(img), 'L')
    #     im.save(f'{output_dir}/{imgs[i].split("/")[-1].split(".")[0]}.jpg')

def create_pair(person_folder, cloth_folder, data_folder):
    print('~~~~~~start the create_pair function')
    os.makedirs(data_folder, exist_ok=True)
    cloths = os.listdir(cloth_folder)
    persons = os.listdir(person_folder)
    with open(os.path.join(data_folder, "test_pairs.txt"), 'w') as file:
        for cloth in cloths:
            if cloth[-4] != '.':
              continue
            for person in persons:
                if person[-4] != '.':
                  continue
                file.write(f"{person} {cloth}\n")

    print('~~~~~~close the create_pair function')

def validation(target_dir):
    path = pathlib.Path(__file__).parent.resolve()
    path1 = os.path.join(path, target_dir, 'test')
    for subfolder in os.listdir(path1):
        cur_dir = os.path.join(path1, subfolder)
        if os.path.isfile(cur_dir):
            os.remove(cur_dir)
        for file in os.listdir(cur_dir):
            if os.path.isdir(os.path.join(cur_dir, file)):
                shutil.rmtree(os.path.join(cur_dir, file))

def try_on(person_folder, cloth_folder, data_folder):

    print('~~~~~~start the try_on function')

    validation(data_folder)
    
    os.system(f"cp {person_folder}/* {data_folder}/test/cloth")
    os.system(f"cp {person_folder}/* {data_folder}/test/cloth-mask")

    cmd = f"python test_generator.py --occlusion --cuda True --tocg_checkpoint checkpoints/mtviton.pth \
            --gpu_ids 0 --gen_checkpoint checkpoints/gen.pth --datasetting unpaired --dataroot {data_folder} \
            --output_dir {data_folder}/results"
    os.system(cmd)

    print('~~~~~~close the try_on function')


def get_cloth(input_path, output_path):
    print('~~~~~~start the get_cloth function')
    path = pathlib.Path(__file__).parent.resolve()
    output_dir = os.path.join(output_path, "cloth")
    os.makedirs(output_dir, exist_ok=True)
    images_list = os.listdir(os.path.join(path, input_path))
    filtered = filter(lambda score: score[-4] == '.', images_list)
    images_list = list(filtered)
    pbar = tqdm(total=len(images_list))
    for image_path in images_list:
        image_path = os.path.join(input_path, image_path)
        np_img = cv2.imread(image_path)
        img = Image.open(image_path).convert("RGB")
        image_tensor = transform_rgb(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        output_tensor = net(image_tensor.to(DEVICE))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()
        
        mask = np.ones_like(np_img) * 255
        mask[output_arr == 2] = np_img[output_arr == 2]
        image_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_dir, image_name[:-4] + "_cloth.jpg"), mask)
        pbar.update(1)
    pbar.close()
    print('~~~~~~close the get_cloth function')

tmp_folder = output_folder = "data"

# input_person = "/root/workspace/source/HR-VITON/data/test/image"
# input_cloth = "/root/workspace/source/HR-VITON/data/test/cloth"

input_person = "data/image"
input_cloth = "data/cloth"

# input_person1 = "test/person_image/1.jpg"

# import time

if os.path.isfile(input_person):
    person_folder = os.path.join(tmp_folder, "image")
    os.makedirs(person_folder, exist_ok = True)
    os.system(f"cp {input_person} {person_folder}/")
else:
    person_folder = input_person

cloth_folder = input_cloth
    

# # t1 = time.time()

# create_pair(person_folder, input_cloth, tmp_folder)
# resize_image(person_folder, output_folder)
# openpose(output_folder, output_folder)
# densepose(output_folder, output_folder)
# get_cloth(input_person, output_folder)
# human_parsing(output_folder, output_folder)
# parse_agnostic(output_folder, output_folder)
# human_agnostic(output_folder, output_folder)

# crop_cloth(input_cloth, output_folder)
# resize_cloth(input_cloth, output_folder)
rm_cloth_background(cloth_folder, output_folder)

# try_on(person_folder, cloth_folder, tmp_folder)
# print(time.time() - t1)
# os.system("rm -rf tmp")

