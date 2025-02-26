NUM_CHANNELS = 3
N_CLASSES = 1
TRAIN_SIZE = 400
LABEL_SIZE = 400
TEST_SIZE = 608

DATADIR = "../Datasets/training/"
IMAGE_DIR = DATADIR + "images/"
MASK_DIR = DATADIR + "groundtruth/"

TEST_SET = "../Datasets/test_set_images/"
DEST_DIR = "../predictions/"

MODELS_DIR = "./models"
MASK_THRESHOLD = 0.25

ROTATION_ANGLES = [0, 15, 30, 45, 60, 75]

SAVE_EVERY = 2000

AUGMENTATION_PROB = 0.25
