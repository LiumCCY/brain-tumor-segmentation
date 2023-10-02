
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config

# Load data
def load_(root):
    Data = pd.read_csv(root, delimiter= ',', encoding= 'utf-8', header= None)
    data = Data.to_numpy()
    trainloss = []
    validloss = []
    trainPCC = []
    validPCC = []
    r2 = []
    
    for i in range(0, np.size(data, axis= 1)):
        trainloss.append(data[0][i])
        validloss.append(data[1][i])
        trainPCC.append(data[2][i])
        validPCC.append(data[3][i])
        r2.append(data[4][i])
    
    return trainloss, validloss, trainPCC, validPCC, r2

# Plot loss
def plotloss(train, valid):
    print(len(train), len(valid))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(0, len(train)), train, color = 'blue', label = 'training loss')
    plt.plot(range(0, len(valid)), valid, color = 'red', label = 'validation loss')
    plt.legend()
    plt.show()

# Plot accuracy
def plotaccu(train, valid):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(range(0, len(train)), train, color = 'blue', label = 'training accuracy')
    plt.plot(range(0, len(valid)), valid, color = 'red', label = 'validation accuracy')
    plt.legend()
    plt.show()

def plot(tl, vl, ta, va):

    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(0, len(tl)), tl, color = 'blue', label = 'training loss')
    plt.plot(range(0, len(vl)), vl, color = 'red', label = 'validation loss')
    plt.legend()

    
    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(range(0, len(ta)), ta, color = 'blue', label = 'training accuracy')
    plt.plot(range(0, len(va)), va, color = 'red', label = 'validation accuracy')
    plt.legend()

    plt.show()
    plt.savefig(config.PREDICT_PATHS)
    
def plot_from_csv(csv_file):
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    train_iou=[]
    val_iou=[]

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # 從資料列中提取數據
    for i in range(len(rows[0])):
        value_str = rows[0][i]
        start_index = value_str.find('(') + 1
        end_index = value_str.find(',')
        value = float(value_str[start_index:end_index])
        train_loss.append(value)

        value_str2 = rows[1][i]
        start_index2 = value_str2.find('(') + 1
        end_index2 = value_str2.find(',')
        value2 = float(value_str2[start_index2:end_index2])
        val_loss.append(value2)
        
        value3 = float(rows[2][i])
        train_accuracy.append(value3)
        
        value4 = float(rows[3][i])
        val_accuracy.append(value4)
        
        value_str5 = rows[4][i]
        start_index5 = value_str5.find('(') + 1
        end_index5 = value_str5.find(',')
        value5 = float(value_str5[start_index5:end_index5])
        train_iou.append(value5)

        value_str6 = rows[5][i]
        start_index6 = value_str6.find('(') + 1
        end_index6 = value_str6.find(',')
        value6 = float(value_str6[start_index6:end_index6])
        val_iou.append(value6)

    plt.figure(figsize=((20,8)))
    plt.subplot(1, 3, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(0, len(train_loss)), train_loss, color='blue', label='train loss')
    plt.plot(range(0, len(val_loss)), val_loss, color='red', label='val loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(-1,1)
    plt.plot(range(0, len(train_accuracy)), train_accuracy, color='blue', label='train PCC')
    plt.plot(range(0, len(val_accuracy)), val_accuracy, color='red', label='val PCC')
    plt.legend()
    plt.show()
    
    plt.subplot(1, 3, 3)
    plt.xlabel('Epoch')
    plt.ylabel('Iou')
    plt.ylim(-1,1)
    plt.plot(range(0, len(train_iou)), train_iou, color='blue', label='train iou')
    plt.plot(range(0, len(val_iou)), val_iou, color='red', label='val iou')
    plt.legend()
    plt.show()

    plt.savefig(config.PLOT_PATH)  
    
def main():
    plot_from_csv(config.RECORD_PATH)
    
main()