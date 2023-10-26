import os

DEVICE = "cuda" 

PIN_MEMORY = True if DEVICE == "cuda" else False

INIT_LR = 0.001
NUM_EPOCHS = 150
BATCH_SIZE = 8

INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGTH = 128

BASE_OUTPUT = "/home/b09508004/snap/output"
BASE_MODEL = "/home/b09508004/snap/saved_model"

MODEL_PATH = os.path.join(BASE_MODEL, "UNet_bce_dice_BS32.pth")
PLOT_PATH = os.path.join(BASE_OUTPUT, "UNet_bce_dice_BS32.png")
PREDICT_PATHS = os.path.join("/home/b09508004/snap/output/predictionResUNet_bce_dice")
CHECKPOINT_PATHS = os.path.join(BASE_MODEL, "checkpoint_UNet_bce_dice_BS32.ckpt")
RECORD_PATH = os.path.join(BASE_OUTPUT, "UNet_bce_dice_BS32_record.csv")