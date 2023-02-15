import pandas as pd
import os

def build_names():
    names = []

    for j in range(1, 7):
        for k in range(1, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def write_loss_csv(csv_path, total_loss, loss_states):
    if os.path.exists(csv_path):
        pass
    else:
        pd.DataFrame().to_csv(csv_path)
        
    df = pd.read_csv(csv_path)
    columns = ["Total Loss", "Loss avg"]
    value = total_loss
    for name, val in loss_states.items():
        columns.append(f"{name} value")
        columns.append(f"{name} avg")
        value.append(val.val)
        value.append(val.avg)
    loss_stages = pd.DataFrame([value], columns=columns)
    df = pd.concat([df, pd.DataFrame(loss_stages)], axis=0, ignore_index=True)
    df.to_csv(csv_path, index= False)