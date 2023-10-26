import sys
sys.path.append('/home/b09508004/snap')

from dataset import Mydataset
from transform import train_transform, test_transform, RandomTransform
from torch.utils.data import DataLoader

from model.unet import *
from model.Unet3Plus import *
from model.ResnetUnet import ResUnet

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from utils.score import *
from utils.loss_function import *
from utils.record import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from scipy.stats import pearsonr

import config
from tqdm.auto import tqdm
import logging
import matplotlib.pyplot as plt
import os

'''Check cuda'''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = "1"
print(torch.cuda.is_available())
torch.cuda.set_device(0)
print(torch.__version__)
print(sys.version)

'''Data ready'''
train_img_path = '/home/b09508004/Data/Training/images/images'
train_label_path = '/home/b09508004/Data/Training/labels/labels'

val_img_path = '/home/b09508004/Data/Validation/images/images'
val_label_path = '/home/b09508004/Data/Validation/labels/labels'

train_transforms = RandomTransform(train_transform)
test_transforms = RandomTransform(test_transform)
trainDS = Mydataset(image_dir=train_img_path, label_dir=train_label_path, transform=train_transforms)
valDS =  Mydataset(image_dir=val_img_path, label_dir=val_label_path, transform=test_transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(valDS)} examples in the test set...")

trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=0)
valLoader = DataLoader(valDS, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=0)
trainSteps = len(trainDS) // config.BATCH_SIZE
valSteps = len(valDS) // config.BATCH_SIZE

'''Check image'''
images, labels = next(iter(trainLoader))
for i in range(4):
    image = images[i]
    label = labels[i] 
    #print(image.shape)
    #print(label.shape)
    image = image.permute(1,2,0).numpy()
    label = label.permute(1,2,0).numpy()    
    #print(image.shape)
    #print(label.shape)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(image,cmap="gray")
    plt.title('Image')
    plt.axis('off') 
    plt.subplot(122)
    plt.imshow(label,cmap="gray")
    plt.title('Label')
    plt.axis('off')
    plt.show()
    plt.savefig("/home/b09508004/snap/test/test{}.".format(i))

'''Model'''
unet = UNet(1,1).to(config.DEVICE)
#resnetunet = ResUnet(1).to(config.DEVICE)
unet3plus = UNet_3Plus(1,1).to(config.DEVICE)
model = unet

'''Loss function'''
mse = nn.MSELoss()
bce = BCEWithLogitsLoss()
loss = bce

'''optimizer'''
opt = Adam(model.parameters(), lr=config.INIT_LR,betas=(0.9, 0.999))
scheduler = ReduceLROnPlateau(opt,mode='min', factor=0.1, patience=10, verbose=True, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

trainloss = []
validloss = []
trainPCC = []
validPCC = []
train_acc=[]
valid_acc=[]
train_iou = []
val_iou = []
train_f1 =[] 
val_f1 =[]
best_acc = 0.0
stale = 0

'''Load'''
#trainloss, validloss, trainPCC, validPCC, train_iou, val_iou, train_f1, val_f1= load_(config.RECORD_PATH)
#model, _ = loadmodel(config.CHECKPOINT_PATHS, model, opt, device= config.DEVICE)
'''Learning Rate adjustment'''
#new_learning_rate = 0.00001
#for param_group in opt.param_groups:
#    param_group['lr'] = new_learning_rate

print("[INFO] training the network...")
logging.basicConfig(filename='trainUNet.log', level=logging.INFO, filemode='a')
progress = tqdm(range(config.NUM_EPOCHS))
gradient_accumulation_steps = 4

for epoch in range(config.NUM_EPOCHS):
	model.train()
	
	totalTrainLoss = 0.0
	train_loss = 0.0
	train_PCC = 0.0
	trainf1 = 0.0
	trainiou =0.0

	for batch_idx, (x, y) in enumerate(trainLoader):
		#x = torch.cat([x,x,x],dim=1)
		x = x.to(torch.float32).to(config.DEVICE)
		y = y.to(torch.float32).to(config.DEVICE)
		pred = model(x)

		train_loss = bce_dice_loss(pred,y)
		totalTrainLoss += train_loss 

		trainiou += dice_coefficient(pred,y,0.5)
		trainf1 += f_score(y,pred)

		y_true_flat_np1 = y.flatten().detach().cpu().numpy()
		pred_flat_np1 = pred.flatten().detach().cpu().numpy()
		corr, _ = pearsonr(pred_flat_np1, y_true_flat_np1)  # 忽略p值
		train_PCC += corr

		train_loss.backward()
		if (batch_idx + 1) % gradient_accumulation_steps == 0:
			opt.step()
			opt.zero_grad()
		torch.cuda.empty_cache()
 
	with torch.no_grad():
		model.eval()

		totalValLoss = 0.0
		valid_loss = 0.0
		valid_PCC = 0.0
		validf1 = 0.0
		valiou =0.0

		for x, y in valLoader:
			#x = torch.cat([x,x,x],dim=1)
			x = x.to(torch.float32).to(config.DEVICE) 
			y = y.to(torch.float32).to(config.DEVICE)
   
			pred = model(x)
	
			val_loss = bce_dice_loss(pred,y)
			totalValLoss += val_loss
			
			valiou += dice_coefficient(pred,y,0.5)
			validf1 += f_score(y,pred)

			y_true_flat_np = y.flatten().cpu().numpy()
			pred_flat_np = pred.flatten().cpu().numpy()
			corr, _ = pearsonr(pred_flat_np, y_true_flat_np)  # 忽略p值
			valid_PCC += corr
			
	avgTrainLoss = totalTrainLoss / trainSteps
	trainloss.append(avgTrainLoss)
	avgValLoss = totalValLoss / valSteps
	validloss.append(avgValLoss)
	avgTrainPCC = train_PCC/trainSteps
	trainPCC.append(avgTrainPCC)
	avgValPCC = valid_PCC/valSteps
	validPCC.append(avgValPCC)
	avgTrainiou = trainiou/trainSteps
	train_iou.append(avgTrainiou)
	avgValiou = valiou/valSteps
	val_iou.append(avgValiou)
	avgTrainf1 = trainf1/trainSteps
	train_f1.append(avgTrainiou)
	avgValf1 = validf1/valSteps
	val_f1.append(avgValiou)

	scheduler.step(avgValLoss)

	avgValiou = torch.tensor(avgValiou) 
	if avgValiou.item() > best_acc:
		print(f"Best model found at epoch {epoch}, saving model")
		torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, config.CHECKPOINT_PATHS)
		torch.save(model, config.MODEL_PATH)
		best_acc = avgValiou
		stale = 0 
	else:
		stale += 1
		if stale >= 20:
			print(f"No improvment 20 consecutive epochs, early stopping")
			break

	savestatis(trainloss, validloss, trainPCC, validPCC, train_iou, val_iou, train_f1, val_f1)
	tqdm.write("[INFO] EPOCH: {}/{}".format(epoch + 1, config.NUM_EPOCHS))
	tqdm.write("Train loss: {:.4f}, Val loss: {:.4f}".format(avgTrainLoss, avgValLoss))
	tqdm.write("Train PCC: {:.4f}, val PCC: {:.4f}".format(avgTrainPCC, avgValPCC))
	tqdm.write("Train iou: {:.4f}, Val iou: {:.4f}".format(avgTrainiou, avgValiou))
	tqdm.write("Train F1: {:.4f}, Val F1: {:.4f}".format(avgTrainf1, avgValf1))
	current_lr = scheduler.optimizer.param_groups[0]['lr']
	logging.info(f"Epoch {epoch + 1}/{epoch}, Train Loss: {avgTrainLoss}, Val Loss: {avgValLoss}, Train Accuracy: {avgTrainPCC}, Val Accuracy: {avgValPCC}, Train iou: {avgTrainiou}, Val iou: {avgValiou}, Train F1: {avgTrainf1}, Val F1: {avgValf1},Learning Rate: {current_lr}")
	progress.update(1)
