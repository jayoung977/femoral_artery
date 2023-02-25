from cmath import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torchvision.models as models

import argparse

import numpy as np
import os


from Dataset import Dataset
# from Lower_Dataset_640_480 import Lower_Dataset
# from None_Vessel_Dataset import None_Vessel_Dataset

import natsort
from torchsummary import summary

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import random

import target 
from torchmetrics import ConfusionMatrix
import shutil
import pandas as pd
import seaborn as sns

def main(args):
    print('gpu_num: ',args.gpu_num)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    torch.set_num_threads(2)
    print('-------<target>-------')
    print(target.target_name)
    print('num_class: ', len(target.target_class))
    print('----------------------')
    ### Train, test ###

    root_path = "/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Final_Data/Severance_Data/Test_Data/Y003_bmode/"
    all_paths = [ root_path + class_fol +'/'+ path for class_fol in os.listdir(root_path) for path in os.listdir(root_path + class_fol) if 'png' in path ] 
    # print(all_paths[0])


    random.seed(42)
    random.shuffle(all_paths)

    trnsfm = transforms.Compose([transforms.ToTensor(), transforms.Resize((240, 320))])

    test_dataset = Dataset(all_paths, trnsfm, False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


    num_classes = len(target.target_class)


    model = models.efficientnet_b3(pretrained=True)


    # print(model)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1]= torch.nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
    model.to(device)


    criterion = nn.CrossEntropyLoss()
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



    #load model
    last_model_root_path = "/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Model_Save/"+args.Save_Path.split('/')[-4] +'/' + target.target_name +'/'
    model_file = os.listdir(last_model_root_path)[0]
    last_model_saved = last_model_root_path + model_file
    print('last_model_saved: ',last_model_saved)
    model_data = torch.load(last_model_saved, map_location='cpu')
    model.load_state_dict(model_data['model_state_dict'])


    # last_model_saved = "/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Model_Save/efficientnet_pretrain/Main_Vessel_All/Best_ACC_0.86410 at Epoch_81_Model.tar"
    # model_data = torch.load(last_model_saved, map_location='cpu')
    # model.load_state_dict(model_data['model_state_dict'])
    f_name = args.Save_Path.split("/")[-2] +'/'
    if os.path.isdir(args.Save_Path.replace(f_name,'')) != True:
        os.mkdir(args.Save_Path.replace(f_name,''))
    if os.path.isdir(args.Save_Path) != True:
        os.mkdir(args.Save_Path)
    Result_Save_Path = args.Save_Path  + target.target_name +'_result/'
    print('Result_Save_Path: ',Result_Save_Path)
    if os.path.isdir(Result_Save_Path) != True:
        os.mkdir(Result_Save_Path)

    confusion_mtx_Save_Path = Result_Save_Path +'confusion_matrix/'
    # print('confusion_mtx_Save_Path: ',confusion_mtx_Save_Path)
    if os.path.isdir(confusion_mtx_Save_Path) != True:
        os.mkdir(confusion_mtx_Save_Path)


    test_total_batch = len(test_loader)


    #클래스별 예측 
    classes = target.target_class_names
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    total_label = []
    total_output = []
    with torch.no_grad():
        model.eval()

        test_loss_sum = 0 
        test_acc_sum = 0
        test_total = 0
        for Test_Iter, (test_img, test_label, file_path) in enumerate(test_loader):                
            X = test_img.to(device)
            Y = test_label.to(device, dtype = torch.int64)


            prediction = model(X)
            if num_classes == 2:
                Y_bice = F.one_hot(Y, num_classes=2)
                test_loss = criterion(prediction, Y_bice.to(torch.float32))
            else: 
                test_loss = criterion(prediction, Y)

            test_loss_sum += test_loss

            _, predicted = torch.max(prediction, 1) 

            test_total += prediction.size(0)
            test_acc_sum += torch.sum(predicted == Y).sum().item() 

            #클래스별 예측 
            for label, prediction,file_path in zip(Y, predicted, file_path):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                else :
                    save_path = Result_Save_Path + "label_{}_{}_predict_{}_{}_".format(label,classes[label],prediction,classes[prediction]) + file_path.split('/')[-1]
                    shutil.copy(file_path,save_path)

                total_pred[classes[label]] += 1
            

            total_label += Y
            total_output += predicted



            # confmat(preds, target)

    
        test_loss_mean = test_loss_sum / test_total_batch
        test_acc_mean =  test_acc_sum / test_total


    print( "Total_Cross_Entropy_Loss : %.5f, Accuracy : %.5f" % (test_loss_mean, test_acc_mean))

    total_output,total_label = torch.tensor(total_output).to(device) ,torch.tensor(total_label).to(device)
    confmat = ConfusionMatrix(num_classes=num_classes).to(device)
    confusion_mtx = confmat(total_output, total_label)
    # print(confusion_mtx)

    plt.figure(figsize=(15,15))
    df_cm = pd.DataFrame(confusion_mtx.cpu(), index=classes, columns=classes).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d",annot_kws={"size": 20})

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(confusion_mtx_Save_Path +'confusion_matrix.png')


    # 각 분류별 정확도(accuracy)를 출력
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            total_pred[classname] = 1
        accuracy = float(correct_count) / total_pred[classname]
        print(f'    Test Accuracy for class: {classname:14s} is {accuracy:.5f} %')
    

 





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', type=str, default="4", help="GPU number configuration")
    parser.add_argument('--batch_size', type=int, default=128, help="GPU number configuration")

    parser.add_argument('--Learning_rate', type=float, default="0.001", help="GPU number configuration")
    parser.add_argument('--Epoch', type=int, default=200, help="GPU number configuration")

    # parser.add_argument('--Save_Path', type=str, default="/home/hankh/Works/Project/Vascular_Ultrasound/JY/Codes/220919_view_classification/bmode/experiment/model/", help="GPU number configuration")
    parser.add_argument('--Save_Path', type=str, default="/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Model_Save/efficientnet_pretrain/mismatched_data_and_confusion_matrix/severance_data_Y003/", help="GPU number configuration")

    
    args = parser.parse_args()

    main(args)
