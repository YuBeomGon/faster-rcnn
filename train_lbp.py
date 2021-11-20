import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms

import argparse
from datetime import datetime

import numpy as np
import os
import time
import random
import csv
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import cv2
import albumentations as A
import albumentations.pytorch
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch.optim as optim

from dataset import LbpDataset, train_transforms, val_transforms, test_transforms, collate_fn, get_data
# from visualize import visualize
# from scheduler import CosineAnnealingWarmUpRestarts
from model import fasterrcnn_resnet101_fpn, fasterrcnn_resnet152_fpn, fasterrcnn_resnet201_fpn, FastRCNNPredictor, fasterrcnn_resnet50_fpn
#from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from engine import train_one_epoch, evaluate
from coco_utils import get_coco, get_coco_kp
from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
import utils

# from _utils import warmup_lr_scheduler, reduce_dict
# from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
# import torchvision
# import torchvision.models.detection
# import torchvision.models.detection.mask_rcnn

import warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"  



def get_train_test_list(df_dir) :
    df = pd.read_csv(df_dir)
    df['label_id'] = df.label.apply(lambda x : 1 if 'ASC-US' in x or 'ASC-US with HPV infection' in x 
                                    or 'AS' in x else 0.)
    df = df[df['label_id'] == 1]    
    
    df['xmax'] = df.apply(lambda x : x['xmin'] + x['w'], axis=1)
    df['ymax'] = df.apply(lambda x : x['ymin'] + x['h'], axis=1)
    df['area'] = df.apply(lambda x : x['w'] * x['h'], axis=1)
    df = df[['ID', 'file_name', 'task', 'bbox', 'xmin', 'ymin', 'xmax', 'ymax', 'w', 'h', 'label',
           'occluded','area', 'des', 'cell_type', 'label_id']] 
    
    df_group = df.groupby('file_name')
    df_list = df.file_name.unique()
    train_list, test_list = train_test_split(df_list, test_size=0.25, random_state=42)
    print('total {} train {} test {}'.format(len(df_list), len(train_list), len(test_list)))

    train_list = [get_data(img_id, df_group) for img_id in train_list]
    test_list = [get_data(img_id, df_group) for img_id in test_list]

    print(len(train_list))
    print(len(test_list))   
    
    return train_list, test_list

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    parser.add_argument('--data-path', default='../data/df.csv', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='fastrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-scheduler', default="multisteplr", help='the lr scheduler (default: multisteplr)')
    parser.add_argument('--lr-step-size', default=20, type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-steps', default=[20, 40, 60, 80], nargs='+', type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-gamma', default=0.2, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='trained_models/lbp_trained_resnet50/', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy (default: hflip)')
    parser.add_argument(
        "--sync-bn",
#         dest="sync_bn",
        default=True,
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://192.168.40.242:50019', help='url used to set up distributed training')

    return parser

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print('device id ', args.device)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    # Data loading code
    data_dir = args.data_path
    train_list, test_list = get_train_test_list(data_dir)
    train_dataset = LbpDataset(train_list, transform=train_transforms)
    test_dataset = LbpDataset(test_list, transform=val_transforms)      

    print("Creating data loaders")
    if args.distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=args.world_size)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                      num_replicas=args.world_size)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

#     if args.aspect_ratio_group_factor >= 0:
#         group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
#         train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
#     else:
#         train_batch_sampler = torch.utils.data.BatchSampler(
#             train_sampler, args.batch_size, drop_last=True)

#     data_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_sampler=train_sampler, num_workers=args.workers,
#         collate_fn=utils.collate_fn)
    
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               collate_fn=utils.collate_fn)    

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)
  


    print("Creating model")
    kwargs = {
        "trainable_backbone_layers": args.trainable_backbone_layers
    }
    if 'resnet201' in args.model :
        model = fasterrcnn_resnet201_fpn(pretrained=True, num_classes=2, min_size=2048, max_size=2048)
    
    elif "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
        num_classes = 91
        print('pretrained', args.pretrained)
        print('model', args.model)
#        model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained,
#                                                                  progress=True, **kwargs)
        model = fasterrcnn_resnet50_fpn(pretrained=True,progress=True, min_size=1800, max_size=1800)
    else :
        print('error, no model')
    
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    if args.distributed and args.sync_bn:
        print('sync batchnorm')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
#    optimizer = torch.optim.SGD(
#        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params, lr=0.001)

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                           "are supported.".format(args.lr_scheduler))
#     lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=8, T_mult=2, eta_max=0.1,  T_up=4, gamma=0.5)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return
    next(iter(data_loader))

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if epoch > 10 and epoch % 2 == 0 :
            if args.output_dir:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                    'epoch': epoch
                }
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))

            # evaluate after every epoch
            evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
#     os.environ['MASTER_ADDR'] = '192.168.40.242'
#     os.environ['MASTER_PORT'] = '50019'    
    main(args)