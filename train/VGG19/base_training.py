import argparse
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn

from torch.optim.lr_scheduler import ReduceLROnPlateau

import os, sys
sys.path.append(os.getcwd())

from lib.network.rtpose_vgg import get_model, use_vgg
from lib.datasets import coco, transforms, datasets
from lib.config import update_config
from trainer import train, train_factory
from validator import validate
from config import get_cfg_defaults, update_config
from config.path_cfg import TRAIN_ANNO, VAL_ANNO, TRAIN_PATH, VAL_PATH, VGG19_RESULTS


cfg = get_cfg_defaults()
cfg = update_config(cfg, './config/params.yaml')


DATA_DIR = cfg.COCO2017_DATASET.DIR

ANNOTATIONS_TRAIN = [os.path.join(DATA_DIR, 'annotations', item) for item in ['person_keypoints_train2017.json']]


def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-annotations', default=ANNOTATIONS_TRAIN)
    group.add_argument('--train-image-dir', default=TRAIN_PATH)
    group.add_argument('--val-annotations', default=VAL_ANNO)
    group.add_argument('--val-image-dir', default=VAL_PATH)
    group.add_argument('--pre-n-images', default=8000, type=int,
                       help='number of images to sampe for pretraining')
    group.add_argument('--n-images', default=None, type=int,
                       help='number of images to sample')
    group.add_argument('--duplicate-data', default=None, type=int,
                       help='duplicate data')
    group.add_argument('--loader-workers', default=cfg.TRAINING.BATCH_SIZE, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=cfg.TRAINING.BATCH_SIZE, type=int,
                       help='batch size')
    group.add_argument('--lr', '--learning-rate', default=1., type=float,
                    metavar='LR', help='initial learning rate')
    group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    group.add_argument('--weight-decay', '--wd', default=0.000, type=float,
                    metavar='W', help='weight decay (default: 1e-4)') 
    group.add_argument('--nesterov', dest='nesterov', default=True, type=bool)     
    group.add_argument('--print_freq', default=50, type=int, metavar='N',
                    help='number of iterations to print the training statistics')    
                   
                                         


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_cli(parser)

    parser.add_argument('--stride-apply', default=1, type=int,
                        help='apply and reset gradients every n batches')
    parser.add_argument('--epochs', default=75, type=int,
                        help='number of epochs to train')
    parser.add_argument('--freeze-base', default=0, type=int,
                        help='number of epochs to train with frozen base')
    parser.add_argument('--pre-lr', type=float, default=1e-4,
                        help='pre learning rate')
    parser.add_argument('--update-batchnorm-runningstatistics',
                        default=False, action='store_true',
                        help='update batch norm running statistics')
    parser.add_argument('--square-edge', default=368, type=int,
                        help='square edge of input images')
    parser.add_argument('--ema', default=1e-3, type=float,
                        help='ema decay constant')
    parser.add_argument('--debug-without-plots', default=False, action='store_true',
                        help='enable debug but dont plot')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')                        
    parser.add_argument('--model_path', default='./network/weight/', type=str, metavar='DIR',
                    help='path to where the model saved')                         
    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
        
    return args

args = cli()

print("Loading dataset...")
# load train data
preprocess = transforms.Compose([
        transforms.Normalize(),
        transforms.RandomApply(transforms.HFlip(), 0.5),
        transforms.RescaleRelative(),
        transforms.Crop(args.square_edge),
        transforms.CenterPad(args.square_edge),
    ])
train_loader, val_loader, train_data, val_data = train_factory(args, preprocess, target_transforms=None)


# model
model = get_model(trunk='vgg19')
model = torch.nn.DataParallel(model).cuda()
# load pretrained
use_vgg(model)


# Fix the VGG weights first, and then the weights will be released
for i in range(20):
    for param in model.module.model0[i].parameters():
        param.requires_grad = False

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov)     
                                                                                          
for epoch in range(5):
    # train for one epoch
    train_loss = train(train_loader, model, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(val_loader, model, epoch)  
                                            
# Release all weights                                   
for param in model.module.parameters():
    param.requires_grad = True

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov)          
                                                    
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

best_val_loss = np.inf


model_save_filename = './network/weight/best_pose.pth'
for epoch in range(5, args.epochs):

    # train for one epoch
    train_loss = train(train_loader, model, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(val_loader, model, epoch)   
    
    lr_scheduler.step(val_loss)                        
    torch.save(model.state_dict(), '{}/checkpoint_epoch_{}.pth'.format(VGG19_RESULTS['checkpoints'], epoch))    
    is_best = val_loss<best_val_loss
    best_val_loss = min(val_loss, best_val_loss)
    if is_best:
        torch.save(model.state_dict(), model_save_filename)      
          
