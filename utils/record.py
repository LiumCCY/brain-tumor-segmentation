import config
import csv
import pandas as pd
import torch
import numpy as np

def savestatis(trainloss, validloss, trainPCC, validPCC, train_iou, val_iou, train_f1, val_f1): #trainr, valr):
    with open(config.RECORD_PATH, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(trainloss)
        w.writerow(validloss)
        w.writerow(trainPCC)
        w.writerow(validPCC)
        w.writerow(train_iou)
        w.writerow(val_iou)
        w.writerow(train_f1)
        w.writerow(val_f1)
        

def load_(root):
    Data = pd.read_csv(root, delimiter= ',', encoding= 'utf-8', header= None)
    data = Data.to_numpy()
    trainloss = data[0].tolist()
    validloss = data[1].tolist()
    trainPCC = data[2].tolist()
    validPCC = data[3].tolist()
    train_iou = data[4].tolist()
    val_iou = data[5].tolist()
    train_f1 = data[6].tolist()
    val_f1 = data[7].tolist()
    '''
    trainloss = []
    validloss = []
    trainPCC = []
    validPCC = []
    train_iou = []
    val_iou =[]
    train_f1 = []
    val_f1 = []
    for i in range(0, np.size(data, axis= 1)):
        trainloss.data[0].tolist()
        validloss.append(data[1][i])
        trainPCC.append(data[2][i])
        validPCC.append(data[3][i])
        train_iou.append(data[4][i])   
        val_iou.append(data[5][i])
        train_f1.append(data[6][i])
        val_f1.append(data[7][i])
    '''
    return trainloss, validloss, trainPCC, validPCC, train_iou, val_iou, train_f1, val_f1

def loadmodel(root, model, opt, device):
    checkpoint = torch.load(root, map_location= device)
    model_state, optimizer_state = checkpoint["model"], checkpoint["optimizer"]
    model.load_state_dict(model_state)
    opt.load_state_dict(optimizer_state)
    return model, opt