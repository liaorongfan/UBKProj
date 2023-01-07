from .config import CfgNode

_C = CfgNode()
# Consumers can get config by:
cfg = _C
""" Follow commont steps to config a experiment:
step 1 : prepare dataset
step 2 : build a dataloader
step 3 : build a model
step 4 : set a loss function
step 5 : set a solver 
step 6 : build a trainer 
step 7 : set test metric
"""
# ------------------------------------------- step 1 : dataset config node ---------------------------------------------
_C.DATA = CfgNode()
_C.DATA.ROOT = "datasets"

_C.DATA.TRAIN_DATA = "./dataset/ukb_array_train.npy"

_C.DATA.VALID_DATA = "./dataset/ukb_array_valid.npy"

_C.DATA.TEST_DATA = "./dataset/ukb_array_test.npy"

# ------------------------------------------ step 2 : dataloader config node -------------------------------------------
_C.DATA_LOADER = CfgNode()
# name of dataloader build function
_C.DATA_LOADER.NAME = "single_frame_data_loader"
_C.DATA_LOADER.DATASET = ""
_C.DATA_LOADER.TRANSFORM = "standard_frame_transform"
_C.DATA_LOADER.TRAIN_BATCH_SIZE = 32
_C.DATA_LOADER.VALID_BATCH_SIZE = 32
_C.DATA_LOADER.NUM_WORKERS = 4
_C.DATA_LOADER.SHUFFLE = True
_C.DATA_LOADER.DROP_LAST = True

_C.DATA_LOADER.SECOND_STAGE = CfgNode()
_C.DATA_LOADER.SECOND_STAGE.METHOD = ""
_C.DATA_LOADER.SECOND_STAGE.TYPE = ""

# ------------------------------------------ step 3 : model config node ------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.NAME = "se_resnet50"
_C.MODEL.PRETRAIN = False
_C.MODEL.NUM_CLASS = 5
_C.MODEL.SPECTRUM_CHANNEL = 50
_C.MODEL.RETURN_FEATURE = False

# ------------------------------------------ step 4 : loss config node -------------------------------------------------
_C.LOSS = CfgNode()
_C.LOSS.NAME = "mean_square_error"

# ------------------------------------------ step 5 : solver config node -----------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.NAME = "sgd"
_C.SOLVER.RESET_LR = False
_C.SOLVER.LR_INIT = 0.01
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.BETA_1 = 0.5
_C.SOLVER.BETA_2 = 0.999
_C.SOLVER.SCHEDULER = "multi_step_scale"
_C.SOLVER.FACTOR = 0.1
_C.SOLVER.MILESTONE = [200, 280]

# ------------------------------------------- step 6:  train config node -----------------------------------------------
_C.TRAIN = CfgNode()
_C.TRAIN.TRAINER = "ImageModalTrainer"
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCH = 300
_C.TRAIN.PRE_TRAINED_MODEL = None
_C.TRAIN.RESUME = ""
_C.TRAIN.LOG_INTERVAL = 1
_C.TRAIN.VALID_INTERVAL = 1
_C.TRAIN.SAVE_LAST = False
_C.TRAIN.OUTPUT_DIR = "results"
# ------------------------------------------- step 7:  test config node ------------------------------------------------
_C.TEST = CfgNode()
_C.TEST.TEST_ONLY = False
_C.TEST.FULL_TEST = False
_C.TEST.WEIGHT = ""
_C.TEST.COMPUTE_PCC = True
_C.TEST.COMPUTE_CCC = True
_C.TEST.SAVE_DATASET_OUTPUT = ""
# ======================================================================================================================
