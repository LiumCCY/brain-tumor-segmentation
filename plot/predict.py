import sys
sys.path.append("/home/b09508004/snap")
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from PIL import Image
from transform import specific_normalize_image,zero_normalize_image
from torchvision import transforms
#from transform import test_transform, zero_normalize_image
import torchvision.transforms as transforms

test_transform = transforms.Compose(
    [
        
        transforms.Resize((config.INPUT_IMAGE_HEIGTH, config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485], std=[0.229]),
        #ZScoreNormalize()
    ]
)

def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage, cmap="gray")
	ax[1].imshow(origMask,cmap="gray")
	ax[2].imshow(predMask,cmap="gray")
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	filename = str(sorted(os.listdir('/home/b09508004/Data/Testing/labels/labels'))[i])
	save_path = os.path.join(config.PREDICT_PATHS,filename)
	figure.tight_layout()
	figure.show()
	figure.savefig(save_path, format='png')
 
 
def make_predictions(model, imagePath, input_folder):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it to float data type, and scale its pixel values
		image = Image.open(imagePath).convert("L")
		orig = image.copy()

		image = test_transform(image)
		image = image.numpy()

		filename = sorted(os.listdir(input_folder))[i]
		groundTruthPath = os.path.join('/home/b09508004/Data/Testing/labels/labels',filename)
		
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_WIDTH,config.INPUT_IMAGE_HEIGTH))
		
		image = np.expand_dims(image, 0).astype("float32")
		print(image.shape)
		print(type(image))

		image = torch.from_numpy(image).to(config.DEVICE)
		#image = torch.cat([image, image, image], dim=1)
		#print(image.shape)
		#print(type(image)

		predMask = model(image)[0].squeeze()
		predMask = predMask.cpu().numpy()
		#print(predMask.shape)
		#print(type(predMask))
  
		predMask = predMask*255
		predMask = predMask.astype(np.uint8)

		prepare_plot(orig, gtMask, predMask)
  

print("[INFO] loading up test image paths...")
file_list = "/home/b09508004/Data/Testing/images/images"
for i in range(15):
    imagePaths = os.path.join(file_list,sorted(os.listdir(file_list))[i])
    net = torch.load(config.MODEL_PATH).to(config.DEVICE)
    make_predictions(net, imagePath=imagePaths, input_folder=file_list)