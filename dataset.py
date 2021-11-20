import numpy as np
import os
import pandas as pd
import torch
import albumentations as A
import albumentations.pytorch
import cv2
import math

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
IMAGE_SIZE = 2048
# IMAGE_SIZE= 1024

train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),   
    A.RandomResizedCrop(height=IMAGE_SIZE,width=IMAGE_SIZE,scale=[0.8,1.0],ratio=[0.8,1.2],p=0.8),
    A.pytorch.ToTensor(), 
], p=1.0, bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.8, label_fields=['labels']))    

val_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
    A.pytorch.ToTensor(),     
], p=1.0, bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.8, label_fields=['labels']))    

test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
    A.pytorch.ToTensor(),     
], p=1.0) 

def collate_fn(batch):
    return tuple(zip(*batch))

def get_data(img_id, df_data):
    if img_id not in df_data.groups:
        return dict(image_id=img_id, boxes=list())
    data  = df_data.get_group(img_id)
    boxes = data[['xmin', 'ymin', 'xmax', 'ymax']].values
#     need to check this for pytorch faster rcnn, mask rcnn
# multi-target not supported 
#     labels = data[['label_id']].values
    labels = data['label_id'].values
    size = data[['w', 'h']].values
    area = data[['area']].values
    ID = data[['ID']].values
    return dict(image_id = img_id, boxes = boxes, labels=labels, size=size, area=area, ID=ID)

class LbpDataset(Dataset):
    def __init__(
        self,
        image_list,
        transform=None,
    ):
        self.image_list = image_list
        self.transform = transform
        self.default_path = '/home/NAS/nas4/project_scl/'
        self.default_path = '/home/k8s-storage/cvat-share-pv/'
        self.threshold = 220
        self.image_mean = torch.tensor([0.485, 0.456, 0.406])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        path = self.image_list[index]['image_id']
#         bbox is coco dataformat
#         xmin, ymin, width, height
        boxes = self.image_list[index]['boxes']
        labels = self.image_list[index]['labels']
        size = self.image_list[index]['size']
#         area = self.image_list[index]['area']
        image_id = torch.tensor([index])
#         image_id = self.image_list[index]['ID']
#         print(area)
#         print(type(boxes))

        image = cv2.imread(self.default_path + path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         boxes = np.array(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         print(type(image))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=boxes, labels=labels)
            image = augmentations["image"]
            boxes = augmentations["bboxes"]
            labels = augmentations["labels"]

        image = image/255.
        image = (image - self.image_mean[:, None, None]) / self.image_std[:, None, None]
        
        if len(boxes) == 0 :
            boxes = np.array([[0,0,.01,.01]])
            labels = np.array([0])
            
            
        iscrowd = torch.zeros((len(boxes)), dtype=torch.int64)
        
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.long) 
        target["image_id"] = torch.as_tensor(image_id, dtype=torch.long)
        target["area"] = torch.as_tensor(area, dtype=torch.float32) 
        target["iscrowd"] = iscrowd
#         target["path"] = path
#         target['image_id'] = torch.as_tensor(path,dtype=torch.long)
            
        return image, target

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = train_transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        print(transforms)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["masks"] = masks
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd

        if self.transforms is not None:
            augmentations = self.transforms(image=img, bboxes=boxes, labels=labels)
            image = augmentations["image"]
            boxes = augmentations["bboxes"]    
            labels = augmentations["labels"]
            
        target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["masks"] = masks
# #         target["path"] = path
#         target["size"] = size   
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels,dtype=torch.long)
        target['size'] = torch.as_tensor(size,dtype=torch.float32)

        return img, target

    def __len__(self):
        return len(self.imgs)