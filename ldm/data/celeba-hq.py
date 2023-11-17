from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


class CelebaDataset(data.Dataset):
    def __init__(self,state,arbitrary_mask_percent=0,**args
        ):
        self.state=state
        self.args=args
        self.arbitrary_mask_percent=arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5),
            A.Blur(p=0.05),
            A.ElasticTransform(p=0.05)
            ])
        self.img_path_list=[]
        self.mask_path_list=[]
        
        self.length = 1000
        flag = True
        for idx in range(15):
            bbox_dir='../dataset/CelebAMask-HQ/CelebAMask-HQ-mask-anno/{}/'.format(idx)
            per_dir_file_list=os.listdir(bbox_dir)
            for file_name in per_dir_file_list:
                splitstr = file_name.split('_')
                if splitstr[1] == 'hair.png':
                    self.mask_path_list.append(os.path.join(bbox_dir,file_name))
                    self.img_path_list.append(os.path.join('../dataset/CelebAMask-HQ/CelebA-HQ-img/', str(int(splitstr[0]))+'.jpg'))

                    if len(self.img_path_list) == 1000 and flag:
                        flag = False
                        if state != "train":
                            return
                        else:
                            self.img_path_list=[]
                            self.mask_path_list=[]


        self.length=len(self.img_path_list)

       

    
    def __getitem__(self, index):
        # img_path_p=os.path.join('/sda/home/qianshengsheng/yzy/dataset/CelebAMask-HQ/CelebA-HQ-img/', str(index)+'.jpg')
        # img_path_m=os.path.join('/sda/home/qianshengsheng/yzy/dataset/CelebAMask-HQ/CelebAMask-HQ-mask-anno/{}/'.format(index // 2000), str(index).zfill(5)+'_hair.png')
        img_path_p = self.img_path_list[index]
        img_path_m = self.mask_path_list[index]

        img_p = Image.open(img_path_p).convert("RGB")
        img_m = Image.open(img_path_m).convert("RGB")

   
        ### Get reference image
        W,H = img_p.size
        img_p_np=cv2.imread(img_path_p)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        img_m_np=cv2.imread(img_path_m)
        img_m_np = cv2.cvtColor(img_m_np, cv2.COLOR_BGR2RGB)
        img_m_np = cv2.resize(img_m_np, (W,H))
        mask = np.where(img_m_np > 0, 1, 0)
        ref_image_tensor = mask * img_p_np
        ref_image_tensor = ref_image_tensor.astype(np.uint8)
        ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)

        # W,H = img_p.size
        # img_p_np=cv2.imread(img_path_p)
        # img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        # min_x = W
        # min_y = H
        # max_x = max_y = 0
        # for x in range(W // 2):
        #     for y in range(H // 2):
        #         # 获取像素值
        #         pixel = img_m.getpixel((x, y))
        #         # 如果像素值为1，则更新矩形的最大和最小坐标
        #         if pixel == (0, 0, 0):
        #             min_x = min(min_x, x)
        #             min_y = min(min_y, y)
        #             max_x = max(max_x, x)
        #             max_y = max(max_y, y)
        # ref_image_tensor=img_p_np[2*min_x:2*max_x+1,2*min_y:2*max_y+1,:]
        # ref_image_tensor=self.random_trans(image=ref_image_tensor)
        # ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
        # ref_image_tensor=get_tensor_clip()(ref_image_tensor)



        ### Generate mask
        image_tensor = get_tensor()(img_p)

        # mask_img=np.zeros((H,W))
        # mask_img[0:5,0:5]=1
        # mask_img=Image.fromarray(mask_img)
        # print(mask_img.size, img_m.size)

        img_m = img_m.resize((W,H))
        mask_tensor=1-get_tensor(normalize=False, toTensor=True)(img_m)

        ### Crop square image
        image_tensor_cropped=image_tensor
        mask_tensor_cropped=mask_tensor

        image_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(mask_tensor_cropped)
        mask_tensor_resize[mask_tensor_resize < 0.5] = 0
        mask_tensor_resize[mask_tensor_resize >= 0.5] = 1
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize

        # print(image_tensor_resize.shape, inpaint_tensor_resize.shape, mask_tensor_resize[:1,:,:].shape, ref_image_tensor.shape)
        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize[:1,:,:],"ref_imgs":ref_image_tensor}



    def __len__(self):
        return self.length




a = CelebaDataset(state='train', dataset_dir='dataset/open-images', arbitrary_mask_percent= 0.5, image_size= 512)
a[0]