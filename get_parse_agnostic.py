import json
from os import path as osp
import os

import numpy as np
from PIL import Image, ImageDraw

import argparse

from tqdm import tqdm
def get_im_parse_agnostic(im_parse, pose_data, w=768, h=1024):
    label_array = np.array(im_parse)
    parse_lower = ((label_array == 9).astype(np.float32))
    agnostic = im_parse.copy()
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    return agnostic

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="dataset dir")
    parser.add_argument('--output_path', type=str, help="output dir")

    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path
    
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
