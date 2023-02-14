from yacs.config import CfgNode as CN

_C = CN()

_C.COCO2017_DATASET = CN()
_C.COCO2017_DATASET.TRAIN = './COCO2017/train2017'
_C.COCO2017_DATASET.TEST = './COCO2017/test2017'
_C.COCO2017_DATASET.VAL = './COCO2017/val2017'
_C.COCO2017_DATASET.TRAIN_ANNOTATIONS = './COCO2017/annotations2017/person_keypoints_train2017.json'
_C.COCO2017_DATASET.VAL_ANNOTATIONS = './COCO2017/annotations2017/person_keypoints_val2017.json'

_C.INFANT_DATASET = CN()
_C.INFANT_DATASET.TRAIN = './COCO2017/train2017'
_C.INFANT_DATASET.TEST = './COCO2017/test2017'
_C.INFANT_DATASET.VAL = './COCO2017/val2017'
_C.INFANT_DATASET.TRAIN_ANNOTATIONS = './COCO2017/annotations2017/person_keypoints_train2017.json'
_C.INFANT_DATASET.VAL_ANNOTATIONS = './COCO2017/annotations2017/person_keypoints_val2017.json'

_C.TRAINING = CN()
_C.TRAINING.DATASET = None
_C.TRAINING.EPOCH = 75
_C.TRAINING.PRINT_FREQ = 20
_C.TRAINING.LOG_FOLDER = './Results'
_C.TRAINING.PRETRAINED_WEIGHT = './models/models_weight/mobilenet_sgd_68.848.pth.tar'
_C.TRAINING.TRANSFER_WEIGHT = None

_C.TEMP = CN()
_C.TEMP.PREPARED_DATA = './temp/prepared_data'

_C.RESULTS_PATH = CN()
_C.RESULTS_PATH.VGG19 = './Results/VGG19'
_C.RESULTS_PATH.MOBILENET_V1 = './Results/MobileNet_v1'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()