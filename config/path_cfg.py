from config import get_cfg_defaults, update_config
import os

cfg =  get_cfg_defaults()
cfg =  update_config(cfg, './config/params.yaml')



def select_dataset(dataset):
    if dataset == 'COCO':
        TRAIN_PATH = cfg.COCO2017_DATASET.TRAIN
        VAL_PATH = cfg.COCO2017_DATASET.VAL
        TEST_PATH = cfg.COCO2017_DATASET.TEST
        TRAIN_ANNO = cfg.COCO2017_DATASET.TRAIN_ANNOTATIONS
        VAL_ANNO = cfg.COCO2017_DATASET.VAL_ANNOTATIONS
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