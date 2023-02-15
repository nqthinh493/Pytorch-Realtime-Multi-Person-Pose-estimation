import time
import torch
import torch.backends.cudnn as cudnn


import os, sys
sys.path.append(os.getcwd())
from utils import build_names, AverageMeter, write_loss_csv
from loss import get_loss
from lib.datasets import coco, transforms, datasets

from config import get_cfg_defaults, update_config
from config.path_cfg import VGG19_RESULTS
cfg = get_cfg_defaults()
cfg = update_config(cfg, './config/params.yaml')

time_flag = cfg.RESULTS_PATH.TIME_FLAG
                                      
def train_factory(args, preprocess, target_transforms):
    cudnn.benchmark = True
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


def train(train_loader, model, optimizer, epoch):
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
        if i % cfg.PRINT_FREQ == 0:
            
            
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader))
            print_string +='Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format( data_time=data_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses)
            
            for i, (name, value) in enumerate(meter_dict.items()):
                if i%2==1:
                    print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\n'.format(name=name, loss=value)
                else:
                    print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            print(print_string)
 
            csv_path = VGG19_RESULTS['CSV_logs'] + '/train_{name}.csv'.format(name = time_flag)
            total_loss = [losses.val.item(), losses.avg.item()]
            write_loss_csv(csv_path, total_loss, meter_dict)
    return losses.avg  
        
        
