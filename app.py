from flask import Flask, request
import glob
import cv2
import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
import numpy as np
import time

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from cloth_segmentation.data.base_dataset import Normalize_image
from cloth_segmentation.utils.saving_utils import load_checkpoint_mgpu
from cloth_segmentation.networks.u2net import U2NET
import base64
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface

from rembg import remove

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

checkpoint_path = "cloth_segmentation/trained_checkpoint/cloth_segm_u2net_latest.pth"
transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(DEVICE)
net = net.eval()

data_folder = "database"

def get_img_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 16).astype(np.float32) +
                    (parse_array == 17).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 19).astype(np.float32))

    
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    r = int(length_a / 16) + 1
    
    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic


def get_im_parse_agnostic(im_parse, pose_data, w=768, h=1024):
    label_array = np.array(im_parse)
    parse_upper = ((label_array == 5).astype(np.float32) +
                    (label_array == 6).astype(np.float32) +
                    (label_array == 7).astype(np.float32))
    # parse_neck = (label_array == 10).astype(np.float32)

    r = 10
    agnostic = im_parse.copy()

    # mask arms
    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', (w, h), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
            pointx, pointy = pose_data[i]
            radius = r*4 if i == pose_ids[-1] else r*15
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i
        parse_arm = (np.array(mask_arm) / 255) * (label_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    # agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    return agnostic

def resize_image(img):
    dst_h = 1024
    dst_w = 768

    h, w, _ = img.shape[:]
    ratio_h = dst_h / h 
    ratio_w = dst_w / w 
    ratio = min(ratio_h, ratio_w)
    img = cv2.resize(img, fx = ratio, fy = ratio, dsize = None)
    h, w, _ = img.shape[:]
    top = (dst_h - h) // 2
    bottom = (dst_h - h) // 2 + (dst_h - h) % 2
    left = (dst_w - w) // 2
    right = (dst_w - w) // 2 + (dst_w - w) % 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)

    return img

def openpose(input_file):

    os.makedirs("tmp_", exist_ok = True)
    os.system(f"cp {input_file} tmp_")

    cmd = f"./openpose/openpose.bin --image_dir tmp_ --hand --disable_blending --display 0 --write_json \
            {data_folder}/openpose_json --write_images {data_folder}/openpose_img --model_folder openpose/models"
    os.system(cmd)
    os.system(f"rm -rf tmp_")
    
def densepose(input_file):
    import sys
    sys.path.append("DensePose")
    os.makedirs("tmp_", exist_ok = True)
    os.system(f"cp {input_file} tmp_")
    cmd = f"python DensePose/apply_net.py show DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
            https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
            tmp_ dp_segm -v --output_dir {data_folder}/image-densepose/"
    os.system(cmd)
    os.system(f"rm -rf tmp_")

def human_parsing(input_file, dataset = "lip", output_dir = "image-parse-v3"):
    import sys
    sys.path.append("Self-Correction-Human-Parsing")
    os.makedirs("tmp_", exist_ok = True)
    os.system(f"cp {input_file} tmp_")
    cmd = f"python Self-Correction-Human-Parsing/simple_extractor.py --dataset {dataset} \
            --model-restore Self-Correction-Human-Parsing/checkpoints/{dataset}.pth --input-dir tmp_ --output-dir {data_folder}/{output_dir}"
    os.system(cmd)
    os.system(f"rm -rf tmp_")

def parse_agnostic(input_file):
    output_path = os.path.join(data_folder, "image-parse-agnostic-v3.2")
    
    os.makedirs(output_path, exist_ok=True)

    image_name = os.path.basename(input_file)
    # load pose image
    pose_name = image_name.replace('.jpg', '_keypoints.json')
    
    try:
        with open(osp.join(data_folder, 'openpose_json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
    except IndexError:
        print(pose_name)

    # load parsing image
    parse_name = image_name.replace('.jpg', '.png')
    im_parse = Image.open(osp.join(data_folder, 'image-parse-v3', parse_name))

    agnostic = get_im_parse_agnostic(im_parse, pose_data)
    
    agnostic.save(osp.join(output_path, parse_name))

def human_agnostic(input_file):
    
    output_path = os.path.join(data_folder, "agnostic-v3.2")
    
    os.makedirs(output_path, exist_ok=True)

    im_name = os.path.basename(input_file)
    # load pose image
    pose_name = im_name.replace('.jpg', '_keypoints.json')
    
    try:
        with open(osp.join(data_folder, 'openpose_json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
    except IndexError:
        print(pose_name)

    # load parsing image
    im = Image.open(osp.join(data_folder, 'image', im_name))
    label_name = im_name.replace('.jpg', '.png')
    im_label = Image.open(osp.join(data_folder, 'image-parse-v3', label_name))
    agnostic = get_img_agnostic(im, im_label, pose_data)
    agnostic.save(osp.join(output_path, im_name))

def rm_cloth_background(img):
    
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(RGB_img)
    
    image_tensor = transform_rgb(pil_img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor.to(DEVICE))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    mask = np.zeros_like(output_arr)
    mask[output_arr == 1] = 255

    mask_img = img.copy()
    mask_img[output_arr == 1] = (0, 0, 0)
    pil_img_mask = Image.fromarray(mask_img)
    bgr_color = pil_img_mask.getcolors(pil_img_mask.size[0]*pil_img_mask.size[1])
    bgr_color = list(filter(lambda x: x[1] != (0, 0, 0), bgr_color))
    bgr_color = max(bgr_color)
    mask_img[:, :, :] = bgr_color[1]
    mask_img[output_arr == 1] = img[output_arr == 1]
    
    return mask_img, mask
    

def try_on_process(data_folder):
    cmd = f"python test_generator.py --occlusion --cuda True --tocg_checkpoint checkpoints/mtviton.pth \
            --gpu_ids 0 --gen_checkpoint checkpoints/gen.pth --datasetting unpaired --dataroot {data_folder} \
            --output_dir {data_folder}/results"
    os.system(cmd)

app = Flask(__name__)

@app.route('/add-cloth', methods=['POST'])
def add_cloth():
    uploaded_file = request.files['file']
    img = Image.open(uploaded_file).convert('RGB')
    img = np.array(img)
    resized_img = resize_image(img)

    cloth_folder = os.path.join(data_folder, "cloth")
    cloth_mask_folder = os.path.join(data_folder, "cloth-mask")
    os.makedirs(cloth_folder, exist_ok=True)
    os.makedirs(cloth_mask_folder, exist_ok=True)

    im = Image.fromarray(resized_img)
    image = interface([im])[0]
    img = np.array(image)
    RGB = img[...,:3] # no transparency
    BGR = RGB[...,::-1]
    idx = (RGB[...,0]==130)&(RGB[...,1]==130)&(RGB[...,2]==130) # background 0 or 130, just try it
    mask = np.ones(idx.shape)*255
    mask[idx] = 0
    BGR[idx==True] = (255, 255, 255)    

    # img_name = f"{int(time.time() * 1e6)}.jpg"
    img_name = os.path.splitext(uploaded_file.filename)[0] + ".jpg"
    
    file_path = os.path.join(cloth_folder, img_name)
    file_mask_path = os.path.join(cloth_mask_folder, img_name)
    cv2.imwrite(file_path, BGR)
    cv2.imwrite(file_mask_path, mask)

    return "Done"

@app.route('/add-cloth-from-model', methods=['POST'])
def add_cloth_from_model():
    uploaded_file = request.files['file']
    img = Image.open(uploaded_file).convert('RGB')
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    resized_img = resize_image(img)

    cloth_folder = os.path.join(data_folder, "cloth")
    cloth_mask_folder = os.path.join(data_folder, "cloth-mask")
    os.makedirs(cloth_folder, exist_ok=True)
    os.makedirs(cloth_mask_folder, exist_ok=True)
    cloth_img, mask = rm_cloth_background(resized_img)

    # img_name = f"{int(time.time() * 1e6)}.jpg"
    img_name = os.path.splitext(uploaded_file.filename)[0] + ".jpg"
    
    file_path = os.path.join(cloth_folder, img_name)
    file_mask_path = os.path.join(cloth_mask_folder, img_name)
    cv2.imwrite(file_path, cloth_img)
    cv2.imwrite(file_mask_path, mask)

    return "Done"

@app.route('/delete-cloth', methods=['GET'])
def delete_cloth():
    content = request.json
    cloth_name = content['cloth']

    cloth_path = os.path.join(data_folder, "cloth", cloth_name)
    cloth_mask_path = os.path.join(data_folder, "cloth-mask", cloth_name)

    os.system(f"rm {cloth_path}")
    os.system(f"rm {cloth_mask_path}")
    return "Done"

@app.route('/delete-all-cloth', methods=['GET'])
def delete_all_cloth():

    cloth_folder = os.path.join(data_folder, "cloth")
    cloth_mask_folder = os.path.join(data_folder, "cloth-mask")

    os.system(f"rm -rf {cloth_folder}/*")
    os.system(f"rm {cloth_mask_folder}/*")
    return "Done"

@app.route('/get-cloths', methods=['GET'])
def get_cloths():
    cloth_folder = os.path.join(data_folder, "cloth")
    if not os.path.exists(cloth_folder):
        return []
    list_imgs = os.listdir(cloth_folder)
    return list_imgs

@app.route('/add-model', methods=['POST'])
def add_model():
    uploaded_file = request.files['file']

    img = Image.open(uploaded_file).convert('RGB')
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model_folder = os.path.join(data_folder, "image")
    os.makedirs(model_folder, exist_ok=True)

    resized_img = resize_image(img)

    bgr_removed = remove(resized_img, bgcolor=(255, 255, 255, 0))

    # img_name = f"{int(time.time() * 1e6)}.jpg"
    img_name = os.path.splitext(uploaded_file.filename)[0] + ".jpg"
    
    file_path = os.path.join(model_folder, img_name)
    cv2.imwrite(file_path, bgr_removed)

    openpose(file_path)
    densepose(file_path)
    human_parsing(file_path)
    parse_agnostic(file_path)
    human_agnostic(file_path)

    return "Done"

@app.route('/delete-model', methods=['GET'])
def delete_model():
    content = request.json
    model_name = content['model']

    image_parse = model_name.replace("jpg", "png")
    openpose_img = model_name.replace(".jpg", "_rendered.png")
    openpose_json = model_name.replace(".jpg", "_keypoints.json")

    os.system(f"rm {data_folder}/agnostic-v3.2/{model_name}")
    os.system(f"rm {data_folder}/image/{model_name}")
    os.system(f"rm {data_folder}/image-densepose/{model_name}")
    os.system(f"rm {data_folder}/image-parse-agnostic-v3.2/{image_parse}")
    os.system(f"rm {data_folder}/image-parse-v3/{image_parse}")
    os.system(f"rm {data_folder}/openpose_img/{openpose_img}")
    os.system(f"rm {data_folder}/openpose_json/{openpose_json}")

    return "Done"

@app.route('/delete-all-model', methods=['GET'])
def delete_all_model():

    os.system(f"rm -rf {data_folder}/agnostic-v3.2/*")
    os.system(f"rm -rf {data_folder}/image/*")
    os.system(f"rm -rf {data_folder}/image-densepose/*")
    os.system(f"rm -rf {data_folder}/image-parse-agnostic-v3.2/*")
    os.system(f"rm -rf {data_folder}/image-parse-v3/*")
    os.system(f"rm -rf {data_folder}/openpose_img/*")
    os.system(f"rm -rf {data_folder}/openpose_json/*")

    return "Done"

@app.route('/get-models', methods=['GET'])
def get_models():
    model_folder = os.path.join(data_folder, "image")
    if not os.path.exists(model_folder):
        return []
    list_imgs = os.listdir(model_folder)
    return list_imgs

@app.route('/try-on', methods=['GET', 'POST'])
def try_on():
    content = request.json
    cloth_name = content['cloth']
    model_name = content['model']

    os.makedirs("tmp/test/agnostic-v3.2", exist_ok=True)
    os.makedirs("tmp/test/cloth", exist_ok=True)
    os.makedirs("tmp/test/cloth-mask", exist_ok=True)
    os.makedirs("tmp/test/image", exist_ok=True)
    os.makedirs("tmp/test/image-densepose", exist_ok=True)
    os.makedirs("tmp/test/image-parse-agnostic-v3.2", exist_ok=True)
    os.makedirs("tmp/test/image-parse-v3", exist_ok=True)
    os.makedirs("tmp/test/openpose_img", exist_ok=True)
    os.makedirs("tmp/test/openpose_json", exist_ok=True)

    image_parse = model_name.replace("jpg", "png")
    openpose_img = model_name.replace(".jpg", "_rendered.png")
    openpose_json = model_name.replace(".jpg", "_keypoints.json")

    os.system(f"cp {data_folder}/agnostic-v3.2/{model_name} tmp/test/agnostic-v3.2")
    os.system(f"cp {data_folder}/cloth/{cloth_name} tmp/test/cloth")
    os.system(f"cp {data_folder}/cloth-mask/{cloth_name} tmp/test/cloth-mask")
    os.system(f"cp {data_folder}/image/{model_name} tmp/test/image")
    os.system(f"cp {data_folder}/image/{model_name} tmp/test/cloth-mask")
    os.system(f"cp {data_folder}/image/{model_name} tmp/test/cloth")
    os.system(f"cp {data_folder}/image-densepose/{model_name} tmp/test/image-densepose")
    os.system(f"cp {data_folder}/image-parse-agnostic-v3.2/{image_parse} tmp/test/image-parse-agnostic-v3.2")
    os.system(f"cp {data_folder}/image-parse-v3/{image_parse} tmp/test/image-parse-v3")
    os.system(f"cp {data_folder}/openpose_img/{openpose_img} tmp/test/openpose_img")
    os.system(f"cp {data_folder}/openpose_json/{openpose_json} tmp/test/openpose_json")

    with open(os.path.join("tmp", "test_pairs.txt"), 'w') as file:
        file.write(f"{model_name} {cloth_name}\n")

    try_on_process("tmp")

    result_path = glob.glob("tmp/results/*")[0]
    with open(result_path, 'rb') as file:
        image_bytes = file.read()
        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
    os.system("rm -rf tmp")

    return base64_image

def get_skin_mask(img_path):
    im_parse = Image.open(img_path)

    label_array = np.array(im_parse)
    parse_skin = ((label_array == 11).astype(np.float32) +
                (label_array == 12).astype(np.float32) +
                (label_array == 13).astype(np.float32) +
                (label_array == 14).astype(np.float32) +
                (label_array == 15).astype(np.float32))
    return parse_skin

@app.route('/try-on-v2', methods=['GET', 'POST'])
def try_on_v2():
    content = request.json
    cloth_name = content['cloth']
    model_name = content['model']

    os.makedirs("tmp/test/agnostic-v3.2", exist_ok=True)
    os.makedirs("tmp/test/cloth", exist_ok=True)
    os.makedirs("tmp/test/cloth-mask", exist_ok=True)
    os.makedirs("tmp/test/image", exist_ok=True)
    os.makedirs("tmp/test/image-densepose", exist_ok=True)
    os.makedirs("tmp/test/image-parse-agnostic-v3.2", exist_ok=True)
    os.makedirs("tmp/test/image-parse-v3", exist_ok=True)
    os.makedirs("tmp/test/openpose_img", exist_ok=True)
    os.makedirs("tmp/test/openpose_json", exist_ok=True)

    image_parse = model_name.replace("jpg", "png")
    openpose_img = model_name.replace(".jpg", "_rendered.png")
    openpose_json = model_name.replace(".jpg", "_keypoints.json")

    os.system(f"cp {data_folder}/agnostic-v3.2/{model_name} tmp/test/agnostic-v3.2")
    os.system(f"cp {data_folder}/cloth/{cloth_name} tmp/test/cloth")
    os.system(f"cp {data_folder}/cloth-mask/{cloth_name} tmp/test/cloth-mask")
    os.system(f"cp {data_folder}/image/{model_name} tmp/test/image")
    os.system(f"cp {data_folder}/image/{model_name} tmp/test/cloth-mask")
    os.system(f"cp {data_folder}/image/{model_name} tmp/test/cloth")
    os.system(f"cp {data_folder}/image-densepose/{model_name} tmp/test/image-densepose")
    os.system(f"cp {data_folder}/image-parse-agnostic-v3.2/{image_parse} tmp/test/image-parse-agnostic-v3.2")
    os.system(f"cp {data_folder}/image-parse-v3/{image_parse} tmp/test/image-parse-v3")
    os.system(f"cp {data_folder}/openpose_img/{openpose_img} tmp/test/openpose_img")
    os.system(f"cp {data_folder}/openpose_json/{openpose_json} tmp/test/openpose_json")

    with open(os.path.join("tmp", "test_pairs.txt"), 'w') as file:
        file.write(f"{model_name} {cloth_name}\n")

    try_on_process("tmp")

    result_path = glob.glob("tmp/results/*")[0]
    name = os.path.basename(result_path)
    human_parsing(result_path, "atr", "../tmp/human_parse_atr")
    result_skin_mask = get_skin_mask(os.path.join(data_folder, "../tmp/human_parse_atr", name))
    # org_skin_mask = get_skin_mask(image_parse)
    org_img = cv2.imread(f"{data_folder}/image/{model_name}")
    result_img = cv2.imread(result_path)
    result_img[result_skin_mask == True] = org_img[result_skin_mask == True]

    cv2.imwrite(result_path, result_img)

    with open(result_path, 'rb') as file:
        image_bytes = file.read()
        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
    os.system("rm -rf tmp")

    return base64_image


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8000)