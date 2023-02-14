import argparse
import time
import torch
# from torch.utils.tensorboard import SummaryWriter

import os, sys
sys.path.append(os.getcwd())
from utils import build_names, AverageMeter
from loss import get_loss
from lib.datasets import coco, transforms, datasets

from config import get_cfg_defaults, update_config
from config.path_cfg import TRAIN_ANNO, VAL_ANNO, TRAIN_PATH, VAL_PATH
cfg = get_cfg_defaults()
cfg = update_config(cfg, './config/params.yaml')
# writer = SummaryWriter(os.path.join(cfg.RESULTS_PATH.VGG19, 'logs'))       


DATA_DIR = '/media/sparc/Data/nqthinh/COCO2017'

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
    group.add_argument('--loader-workers', default=4, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=32, type=int,
                       help='batch size')
    group.add_argument('--lr', '--learning-rate', default=1., type=float,
                    metavar='LR', help='initial learning rate')
    group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    group.add_argument('--weight-decay', '--wd', default=0.000, type=float,
                    metavar='W', help='weight decay (default: 1e-4)') 
    group.add_argument('--nesterov', dest='nesterov', default=True, type=bool)     
    group.add_argument('--print_freq', default=1, type=int, metavar='N',
                    help='number of iterations to print the training statistics')    
                   
                                         
def train_factory(args, preprocess, target_transforms):
    train_datas = [datasets.CocoKeypoints(
        root=args.train_image_dir,
        annFile=item,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    ) for item in args.train_annotations]

    train_data = torch.utils.data.ConcatDataset(train_datas)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    val_data = datasets.CocoKeypoints(
        root=args.val_image_dir,
        annFile=args.val_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return train_loader, val_loader, train_data, val_data

def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_cli(parser)
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
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


def train(train_loader, model, optimizer, epoch, iter):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, heatmap_target, paf_target) in enumerate(train_loader):
        # measure data loading time
        #writer.add_text('Text', 'text logged at step:' + str(i), i)
        
        #for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(),i)        
        data_time.update(time.time() - end)

        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        paf_target = paf_target.cuda()
        # compute output
        _,saved_for_loss = model(img)
        
        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, paf_target)
        
        for name,_ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
        losses.update(total_loss, img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % cfg.TRAINING.PRINT_FREQ == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader))
            print_string +='Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format( data_time=data_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses)
            # scalars = {}
            for j, (name, value) in enumerate(meter_dict.items()):
                # scalars[f"{name}"] = value
                
                if j%2==1:
                    print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\n'.format(name=name, loss=value)
                else:
                    print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            iter = iter + 1
            # writer.add_scalar("Loss Stage", scalars, iter+1)
            print(print_string)
        
        # if i % cfg.TRAINING.CHECKPOINT_FREQ ==0:
            
            
    return losses.avg  
        
        
