import sys
sys.path.append('/home/b09508004/snap')
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd
import config

plt.style.use("classic")
plt.rcParams['font.sans-serif'] = 'STSong'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 200

sns.set_style("whitegrid")

def plot_from_csv(csv_file):
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    train_dice = []
    val_dice = []
    train_f1 = []
    val_f1 = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        
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
        start_index = value_str5.find('(') + 1
        end_index = value_str5.find(',')
        value5 = float(value_str5[start_index:end_index])
        train_dice.append(value5)

        value_str6 = rows[5][i]
        start_index2 = value_str6.find('(') + 1
        end_index2 = value_str6.find(',')
        value6 = float(value_str6[start_index2:end_index2])
        val_dice.append(value6)
        
        value_str7 = rows[6][i]
        start_index = value_str7.find('(') + 1
        end_index = value_str7.find(',')
        value7 = float(value_str7[start_index:end_index])
        train_f1.append(value7)

        value_str8 = rows[7][i]
        start_index2 = value_str8.find('(') + 1
        end_index2 = value_str8.find(',')
        value8 = float(value_str8[start_index2:end_index2])
        val_f1.append(value8)
        
    # 將數據轉換為 Pandas DataFrame
    df = pd.DataFrame({'Train Loss': train_loss, 'Val Loss': val_loss, 'Train PCC': train_accuracy, 'Val PCC': val_accuracy, 'Train dice': train_dice, 'Val dice': val_dice,'Train F1': train_f1, 'Val F1': val_f1 ,'Epoch': range(len(train_loss))})
    print(df.head())
    plt.figure(figsize=((30, 8)))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    
    plt.subplot(1, 4, 1)
    sns.lineplot(data=df, x='Epoch', y='Train Loss', label='Train Loss', linewidth=2, color='blue')
    sns.lineplot(data=df, x='Epoch', y='Val Loss', label='Val Loss',  linewidth=2, color='red')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(1, 4, 2)
    sns.lineplot(data=df, x='Epoch', y='Train PCC', label='Train PCC', color='blue')
    sns.lineplot(data=df, x='Epoch', y='Val PCC', label='Val PCC', color='red')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('PCC', fontsize=16)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')

    plt.subplot(1, 4, 3)
    sns.lineplot(data=df, x='Epoch', y='Train dice', label='Train dice', color='blue')
    sns.lineplot(data=df, x='Epoch', y='Val dice', label='Val dice', color='red')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Dice', fontsize=16)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    
    plt.subplot(1, 4, 4)
    sns.lineplot(data=df, x='Epoch', y='Train F1', label='Train F1', color='blue')
    sns.lineplot(data=df, x='Epoch', y='Val F1', label='Val F1', color='red')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('F1', fontsize=16)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(config.PLOT_PATH) 

plot_from_csv(config.RECORD_PATH)

