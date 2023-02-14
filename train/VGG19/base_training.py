import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


import os, sys
sys.path.append(os.getcwd())
from lib.network.rtpose_vgg import get_model, use_vgg
from lib.datasets import coco, transforms, datasets
from trainer import cli, train_factory, train
from validator import validate

from config import get_cfg_defaults, update_config
from config.path_cfg import TRAIN_ANNO, VAL_ANNO, TRAIN_PATH, VAL_PATH

cfg = get_cfg_defaults()
cfg = update_config(cfg, './config/params.yaml')
writer = SummaryWriter(os.path.join(cfg.RESULTS_PATH.VGG19, 'logs'))       



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
                                                                                          
# for epoch in range(5):
#     # train for one epoch
#     train_loss = train(train_loader, model, optimizer, epoch)

#     # evaluate on validation set
#     val_loss = validate(val_loader, model, epoch)  
                                            
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
iter = 0
for epoch in range(0, args.epochs):
    
    # train for one epoch
    
    train_loss = train(train_loader, model, optimizer, epoch, iter)

    # evaluate on validation set
    val_loss = validate(val_loader, model, epoch, iter)   
    scalars = {"Train loss": train_loss,
               "Val loss": val_loss}
    writer.add_scalar("Loss", scalars, epoch)
    lr_scheduler.step(val_loss)                        
    
    is_best = val_loss<best_val_loss
    best_val_loss = min(val_loss, best_val_loss)
    if is_best:
        torch.save(model.state_dict(), model_save_filename)  
            
writer.flush()

writer.close()

