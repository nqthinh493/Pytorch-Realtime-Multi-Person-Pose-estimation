import os,sys
sys.path.append(os.getcwd())
from config import get_cfg_defaults, update_config


cfg =  get_cfg_defaults()
cfg =  update_config(cfg, './config/params.yaml')


time_flag = cfg.RESULTS_PATH.TIME_FLAG
def select_dataset(dataset):
    if dataset == 'COCO':
        TRAIN_PATH = cfg.COCO2017_DATASET.DIR + '/train2017'
        VAL_PATH = cfg.COCO2017_DATASET.DIR + '/val2017'
        TEST_PATH = cfg.COCO2017_DATASET.DIR + '/test2017'
        TRAIN_ANNO = cfg.COCO2017_DATASET.DIR + '/annotations/person_keypoints_train2017.json'
        VAL_ANNO = cfg.COCO2017_DATASET.DIR + '/annotations/person_keypoints_val2017.json'
        return TRAIN_PATH, VAL_PATH, TEST_PATH, TRAIN_ANNO, VAL_ANNO
    elif dataset == 'INFANT':
        TRAIN_PATH = cfg.INFANT_DATASET.TRAIN
        VAL_PATH = cfg.INFANT_DATASET.VAL
        TEST_PATH = cfg.INFANT_DATASET.TEST
        TRAIN_ANNO = cfg.INFANT_DATASET.TRAIN_ANNOTATIONS
        VAL_ANNO = cfg.INFANT_DATASET.VAL_ANNOTATIONS
        return TRAIN_PATH, VAL_PATH, TEST_PATH, TRAIN_ANNO, VAL_ANNO
    
    else:
        print('Please add dataset and selection in config!!!')
        
dataset = cfg.TRAINING.DATASET
PREPARED_TRAIN = os.path.join(cfg.TEMP.PREPARED_DATA, f'prepared_train_annotation_{dataset}.pkl')
PREPARED_VAL = os.path.join(cfg.TEMP.PREPARED_DATA, f'val_subset_{dataset}.json')
TRAIN_PATH, VAL_PATH, TEST_PATH, TRAIN_ANNO, VAL_ANNO = select_dataset(dataset)

def check_and_creat_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path
        
    
VGG19_checkpoint_path = check_and_creat_folder(cfg.RESULTS_PATH.VGG19 + f"/checkpoints/{time_flag}")
VGG19_CSV_logs_path = check_and_creat_folder(cfg.RESULTS_PATH.VGG19 + f"/CSV_logs/{time_flag}")
VGG19_accuracy_path = check_and_creat_folder(cfg.RESULTS_PATH.VGG19 + f"/Accuracy/{time_flag}")

VGG19_RESULTS = {'checkpoints': VGG19_checkpoint_path,
                 'CSV_logs': VGG19_CSV_logs_path,
                 'Acc': VGG19_accuracy_path}
