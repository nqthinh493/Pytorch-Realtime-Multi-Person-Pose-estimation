import torch
from collections import OrderedDict
import torch.nn as nn

import os, sys
sys.path.append(os.getcwd())
from utils import build_names



def get_loss(saved_for_loss, heat_temp, vec_temp):

    names = build_names()
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(reduction='mean').cuda()
    total_loss = 0

    for j in range(6):
        pred1 = saved_for_loss[2 * j]
        pred2 = saved_for_loss[2 * j + 1] 


        # Compute losses
        loss1 = criterion(pred1, vec_temp)
        loss2 = criterion(pred2, heat_temp) 

        total_loss += loss1
        total_loss += loss2
        # print(total_loss)

        # Get value from Variable and save for log
        saved_for_log[names[2 * j]] = loss1.item()
        saved_for_log[names[2 * j + 1]] = loss2.item()

    saved_for_log['max_ht'] = torch.max(saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data).item()

    return total_loss, saved_for_log