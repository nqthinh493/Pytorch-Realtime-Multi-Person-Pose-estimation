import time

import os, sys
sys.path.append(os.getcwd())
from utils import build_names, AverageMeter, write_loss_csv
from loss import get_loss

from config import get_cfg_defaults, update_config
from config.path_cfg import VGG19_RESULTS
cfg = get_cfg_defaults()
cfg = update_config(cfg, './config/params.yaml')

time_flag = cfg.RESULTS_PATH.TIME_FLAG

def validate(val_loader, model, epoch):
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
    model.eval()

    end = time.time()
    for i, (img, heatmap_target, paf_target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        paf_target = paf_target.cuda()
        
        # compute output
        _,saved_for_loss = model(img)
        
        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, paf_target)
        
        
        #for name,_ in meter_dict.items():
        #    meter_dict[name].update(saved_for_log[name], img.size(0))
            
        losses.update(total_loss.item(), img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()  
        if i % cfg.TRAINING.CHECKPOINT_FREQ == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(val_loader))
            print_string +='Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format( data_time=data_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)
            for name, value in meter_dict.items():
                print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            print(print_string)
    
            csv_path = VGG19_RESULTS['CSV_logs'] + '/val_{name}.csv'.format(name = time_flag)
            total_loss = [losses.val, losses.avg]
            write_loss_csv(csv_path, total_loss, meter_dict)
    return losses.avg




