from cmath import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

import argparse

import numpy as np
import os
from Model import resnet50

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

def main(args):
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
    root_path = "/home/hankh/Works/Project/Vascular_Ultrasound/KH/Dataset/View_Classification/Vessel/"
    all_paths = [ root_path + class_fol +'/'+ path for class_fol in os.listdir(root_path) for path in os.listdir(root_path + class_fol)] 
    # print('all_paths[0]: ', all_paths[0])
    random.seed(42)
    random.shuffle(all_paths)

    trnsfm = transforms.Compose([transforms.ToTensor(), transforms.Resize((240, 320))])

    train_dataset = Dataset(all_paths, trnsfm, True)
    test_dataset = Dataset(all_paths, trnsfm, False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


    num_classes = len(target.target_class)

    model = resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)
    # print(model)

    # model(num_classes)
    criterion = nn.CrossEntropyLoss()
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    Model_Save_Path = args.Save_Path 

    weight_files = [ Model_Save_Path + file_name for file_name in os.listdir(Model_Save_Path)]
    weight_files = natsort.natsorted(weight_files)
    if (len(weight_files)!= 0):
        model_data = torch.load(weight_files[-1])
        model.load_state_dict(model_data['model_state_dict'])
        optimizer.load_state_dict(model_data['optimizer_state_dict'])

        start_epoch = model_data['epoch'] + 1


    else:
        start_epoch = 0



    best_accuracy = 0
    best_accuracy_epoch = 0
    best_loss = inf

    train_total_batch = len(train_loader)
    test_total_batch = len(test_loader)

    for epoch in range(start_epoch, args.Epoch):
        model.train()

        train_loss_sum = 0
        train_acc_sum = 0
        train_total = 0
        for Train_Iter, (train_img, train_label, file_path) in enumerate(train_loader):
            X = train_img.to(device)
            Y = train_label.to(device, dtype = torch.int64)
            # labels = labels.astype(np.float32).tolist()
            # print('len(X): ',len(X))
            optimizer.zero_grad()

            prediction = model(X)
            # print('len(prediction): ',len(prediction))
            # print('prediction.size(0): ',prediction.size(0))
            if num_classes == 2:
                Y_bice = F.one_hot(Y, num_classes=2)
                train_loss = criterion(prediction, Y_bice.to(torch.float32))
            else: 
                train_loss = criterion(prediction, Y)
            train_loss_sum += train_loss

            _, predicted = torch.max(prediction, 1)   
                     
            train_total += prediction.size(0)
            train_acc_sum += torch.sum(predicted == Y).sum().item() 
            # print('Y: ',Y)
            # print('Y.data: ',Y.data)
            # print(np.shape(Image), np.shape(preds), np.shape(preds == Label.data))

            train_loss.backward()
            optimizer.step()

        # scheduler.step()
        train_loss_mean = train_loss_sum / train_total_batch
        train_acc_mean = train_acc_sum / train_total
        print("Training Epoch : %d, Cross_Entropy_Loss : %.5f, Accuracy : %.5f" % (epoch , train_loss_mean, train_acc_mean))

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
        
            test_loss_mean = test_loss_sum / test_total_batch
            test_acc_mean =  test_acc_sum / test_total

        print("Test Epoch : %d, Cross_Entropy_Loss : %.5f, Accuracy : %.5f, Best_Accruacy : %.5f at Epoch %d" % (epoch , test_loss_mean, test_acc_mean, best_accuracy, best_accuracy_epoch))

        Model_Save_Path = Model_Save_Path + target.target_name +'/'
        print('Model_Save_Path: ',Model_Save_Path)
        if os.path.isdir(Model_Save_Path) != True:
            os.mkdir(Model_Save_Path)
        if(test_acc_mean > best_accuracy):
            best_accuracy_epoch = epoch
            best_accuracy = test_acc_mean

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch' : epoch,
            } , os.path.join(Model_Save_Path + "Best_ACC_%.5f at Epoch_%d_Model.tar" % (best_accuracy, epoch)))
            
        if (epoch - best_accuracy_epoch >= 5):
            break
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', type=str, default="1", help="GPU number configuration")
    parser.add_argument('--batch_size', type=int, default=196, help="GPU number configuration")

    parser.add_argument('--Learning_rate', type=float, default="0.001", help="GPU number configuration")
    parser.add_argument('--Epoch', type=int, default=200, help="GPU number configuration")

    parser.add_argument('--Save_Path', type=str, default="/home/hankh/Works/Project/Vascular_Ultrasound/JY/Codes/220919_view_classification/bmode/experiment/model/", help="GPU number configuration")

    parser.add_argument('--epsilon', type=float, default=0.00000001)
    
    args = parser.parse_args()

    main(args)