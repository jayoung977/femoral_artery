from cmath import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch.optim as optim
import torchvision.models as models

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

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def set_parameter_requires_grad_unfreeze(model, where_unfreeze):
    for name, param in model.named_parameters():
        if (where_unfreeze == 1):
            if 'features.8' in name:
                param.requires_grad = True
        elif (where_unfreeze == 2):
            if (('features.8' in name) or ('features.7' in name)):
                param.requires_grad = True
        elif (where_unfreeze == 3):
            if (('features.8' in name) or ('features.7' in name) or ('features.6' in name)):
                param.requires_grad = True
        elif (where_unfreeze == 4):
            if (('features.8' in name) or ('features.7' in name) or ('features.6' in name) or ('features.5' in name)):
                param.requires_grad = True    
        elif (where_unfreeze == 5):
            if (('features.8' in name) or ('features.7' in name) or ('features.6' in name) or ('features.5' in name) or ('features.4' in name)):
                param.requires_grad = True   
        elif (where_unfreeze == 6):
            if (('features.8' in name) or ('features.7' in name) or ('features.6' in name) or ('features.5' in name) or ('features.4' in name) or ('features.3' in name)):
                param.requires_grad = True   
        elif (where_unfreeze == 7):
            if (('features.8' in name) or ('features.7' in name) or ('features.6' in name) or ('features.5' in name) or ('features.4' in name) or ('features.3' in name) or ('features.2' in name)):
                param.requires_grad = True  
        elif (where_unfreeze == 8):
            if (('features.8' in name) or ('features.7' in name) or ('features.6' in name) or ('features.5' in name) or ('features.4' in name) or ('features.3' in name) or ('features.2' in name) or ('features.1' in name)):
                param.requires_grad = True  


# def pretrain_layer(Break_Check,Break_Flag):
#     if (Break_Check in [1,2,3,4,5,6]  and Break_Flag == 5):
#         Break_Check= Break_Check + 1
#         set_parameter_requires_grad_unfreeze(Model, Break_Check)
#         Break_Flag = 1
#         print("##### Breack_Check {} #####".format(Break_Check))
#     elif (Break_Check == 7  and Break_Flag == 25):
#         Break_Check= Break_Check + 1
#         set_parameter_requires_grad_unfreeze(Model, Break_Check)
#         Break_Flag = 1
#         print("##### Breack_Check {} #####".format(Break_Check))
#     elif (Break_Check == 8  and Break_Flag == 5):
#         break

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
    # root_path = "/home/hankh/Works/Project/Vascular_Ultrasound/KH/Dataset/View_Classification/Vessel/"
    root_path = "/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Vessel/"
    all_paths = [ root_path + class_fol +'/'+ path for class_fol in os.listdir(root_path) for path in os.listdir(root_path + class_fol)] 
    # print('all_paths[0]: ', all_paths[0])
    random.seed(42)
    random.shuffle(all_paths)

    trnsfm = transforms.Compose([transforms.ToTensor(), transforms.Resize((240, 320))])

    train_dataset = Dataset(all_paths, trnsfm, True)
    test_dataset = Dataset(all_paths, trnsfm, False)

    #class 0 : 43200개, class 1 : 4800개
    class_name , class_counts,  labels= train_dataset.value_counts()#43200, 4800

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


    num_classes = len(target.target_class)


    model = models.efficientnet_b3(pretrained=True)

    set_parameter_requires_grad(model, feature_extracting=True)


    # print(model)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1]= torch.nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
    model.to(device)


    nSamples = class_counts
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    criterion = nn.CrossEntropyLoss(normedWeights)


    # criterion = nn.CrossEntropyLoss()


    if num_classes == 2:
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.BCEWithLogitsLoss(normedWeights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # Model_Save_Path = args.Save_Path 
    Model_Save_Path = args.Save_Path  + target.target_name +'/'
    print('Model_Save_Path: ',Model_Save_Path)
    if os.path.isdir(Model_Save_Path) != True:
        os.mkdir(Model_Save_Path)

    weight_files = [ Model_Save_Path + file_name for file_name in os.listdir(Model_Save_Path)]
    weight_files_epoch = [ int(file_name.split('Epoch_')[-1].split('_')[0]) for file_name in os.listdir(Model_Save_Path) ]
    # weight_files = natsort.natsorted(weight_files)

    if (len(weight_files)!= 0):
        max_epoch_index = weight_files_epoch.index(max(weight_files_epoch))
        last_model_saved = weight_files[max_epoch_index]
        model_data = torch.load(last_model_saved)
        model.load_state_dict(model_data['model_state_dict'])
        optimizer.load_state_dict(model_data['optimizer_state_dict'])


        start_epoch = model_data['epoch'] + 1
        Break_Check = model_data['Break_Check']
        Break_Flag = model_data['Break_Flag'] 
        set_parameter_requires_grad_unfreeze(model, Break_Check)
        print('start_epoch: ',start_epoch, 'Break_Check: ',Break_Check,'Break_Flag: ',Break_Flag)
        
        if 'Best_ACC' in model_data:
            print('**before model is best model**')
            bf_test_accuracy = 0
            bf_test_accuracy_epoch = 0
            best_accuracy = model_data['Best_ACC']
            best_accuracy_epoch = model_data['epoch']
        else:
            print('**before model is not a best model**')
            bf_test_accuracy = 0 
            bf_test_accuracy_epoch = 0
            best_accuracy = 0
            best_accuracy_epoch = 0


    else:
        print('**before model is not exist**')
        bf_test_accuracy = 0 
        bf_test_accuracy_epoch = 0
        best_accuracy = 0
        best_accuracy_epoch = 0
        start_epoch = 0
        Break_Check = 0
        Break_Flag = 1

    # Break_Check = 0
    # Break_Flag = 1

    # best_accuracy = 0
    # best_accuracy_epoch = 0



    best_loss = inf

    train_total_batch = len(train_loader)
    test_total_batch = len(test_loader)

    for epoch in range(start_epoch, args.Epoch):
        model.train()

        train_loss_sum = 0
        train_acc_sum = 0
        train_total = 0
        train_label_total = []
        for Train_Iter, (train_img, train_label, file_path) in enumerate(train_loader):
            X = train_img.to(device)
            Y = train_label.to(device, dtype = torch.int64)

            train_label_total += Y.cpu()

            optimizer.zero_grad()

            prediction = model(X)

            if num_classes == 2:
                Y_bice = F.one_hot(Y, num_classes=2)
                train_loss = criterion(prediction, Y_bice.to(torch.float32))
            else: 
                train_loss = criterion(prediction, Y)
            train_loss_sum += train_loss

            _, predicted = torch.max(prediction, 1)   
                     
            train_total += prediction.size(0)
            train_acc_sum += torch.sum(predicted == Y).sum().item() 

            train_loss.backward()
            optimizer.step()



        # class_unique, class_per_num = np.unique(np.array(train_label_total),return_counts = True  )
        # print('epoch_per_train_label_total')
        # for uni, num in zip(class_unique, class_per_num):
        #     print(uni,':',num)

        train_loss_mean = train_loss_sum / train_total_batch
        train_acc_mean = train_acc_sum / train_total
        print("Training Epoch : %d, Cross_Entropy_Loss : %.5f, Accuracy : %.5f" % (epoch , train_loss_mean, train_acc_mean))
        
        #클래스별 예측 
        classes = target.target_class_names
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
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
                for label, prediction in zip(Y, predicted):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        
            test_loss_mean = test_loss_sum / test_total_batch
            test_acc_mean =  test_acc_sum / test_total

        print("Test Epoch : %d, Cross_Entropy_Loss : %.5f, Accuracy : %.5f, Best_Accruacy : %.5f at Epoch %d" % (epoch , test_loss_mean, test_acc_mean, best_accuracy, best_accuracy_epoch))
        
        # 각 분류별 정확도(accuracy)를 출력
        for classname, correct_count in correct_pred.items():
            if total_pred[classname] == 0:
                total_pred[classname] = 1
            accuracy = float(correct_count) / total_pred[classname]
            print(f'    Test Accuracy for class: {classname:14s} is {accuracy:.5f} %')
        

        # if (Break_Check != 8 and Break_Flag < 25):
        if (Break_Check < 8)  or (Break_Check == 8  and  Break_Flag < 5 ):

            if(test_acc_mean > best_accuracy)  :
                if (best_accuracy != 0) and (best_accuracy_epoch != 0) :
                    os.remove(Model_Save_Path + "Best_ACC_%.5f at Epoch_%d_Model.tar" % (best_accuracy, best_accuracy_epoch))
                best_accuracy_epoch = epoch
                best_accuracy = test_acc_mean

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch' : epoch,
                    'Best_ACC' : best_accuracy,
                    'Break_Check' : Break_Check,
                    'Break_Flag' : Break_Flag    
                    

                } , os.path.join(Model_Save_Path + "Best_ACC_%.5f at Epoch_%d_Model.tar" % (best_accuracy, epoch)))


            else:
                print('Break_Flag: ', Break_Flag)
                if (bf_test_accuracy_epoch != 0) and (bf_test_accuracy != 0) :
                    os.remove(Model_Save_Path + "Best_NOT_ACC_%.5f at Epoch_%d_Model.tar" % (bf_test_accuracy, bf_test_accuracy_epoch))
                bf_test_accuracy_epoch = epoch
                bf_test_accuracy = test_acc_mean 
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch' : epoch,
                    'Break_Check' : Break_Check,
                    'Break_Flag' : Break_Flag
                    

                } , os.path.join(Model_Save_Path + "Best_NOT_ACC_%.5f at Epoch_%d_Model.tar" % (bf_test_accuracy, bf_test_accuracy_epoch)))
                
                Break_Flag = Break_Flag + 1      
                    
        print("-"*100)

        # if (epoch - best_accuracy_epoch >= 5):
        #     break

        # pretrain_layer(Break_Check,Break_Flag)
                
        if (Break_Check == 0 and Break_Flag == 5):
            Break_Check = 1
            set_parameter_requires_grad_unfreeze(model, Break_Check)
            Break_Flag = 1
            print("#"*100 +"##### Breack_Check 1 #####")

        if (Break_Check == 1 and Break_Flag == 5):
            Break_Check= 2
            set_parameter_requires_grad_unfreeze(model, Break_Check)
            Break_Flag = 1
            print("#"*100 +"##### Breack_Check 2 #####")

        if (Break_Check == 2 and Break_Flag == 5):
            Break_Check= 3
            set_parameter_requires_grad_unfreeze(model, Break_Check)
            Break_Flag = 1
            print("#"*100 +"##### Breack_Check 3 #####")

        if (Break_Check == 3 and Break_Flag == 5):
            Break_Check= 4
            set_parameter_requires_grad_unfreeze(model, Break_Check)
            Break_Flag = 1
            print("#"*100 +"##### Breack_Check 4 #####")

        if (Break_Check == 4 and Break_Flag == 5):
            Break_Check= 5
            set_parameter_requires_grad_unfreeze(model, Break_Check)
            Break_Flag = 1
            print("#"*100 +"##### Breack_Check 5 #####")


        if (Break_Check == 5 and Break_Flag == 5):
            Break_Check= 6
            set_parameter_requires_grad_unfreeze(model, Break_Check)
            Break_Flag = 1
            print("#"*100 +"##### Breack_Check 6 #####")

        if (Break_Check == 6 and Break_Flag == 5):
            Break_Check= 7
            set_parameter_requires_grad_unfreeze(model, Break_Check)
            Break_Flag = 1
            print("#"*100 +"##### Breack_Check 7 #####")

        if (Break_Check == 7 and Break_Flag == 25):
            Break_Check= 8
            set_parameter_requires_grad_unfreeze(model, Break_Check)
            Break_Flag = 1
            print("#"*100 +"##### Breack_Check 8 #####")

        if (Break_Check == 8 and Break_Flag == 5):
            print('&'*200)
            print('final')
            # break


 





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', type=str, default="4", help="GPU number configuration")
    parser.add_argument('--batch_size', type=int, default=128, help="GPU number configuration")

    parser.add_argument('--Learning_rate', type=float, default="0.001", help="GPU number configuration")
    parser.add_argument('--Epoch', type=int, default=200, help="GPU number configuration")

    # parser.add_argument('--Save_Path', type=str, default="/home/hankh/Works/Project/Vascular_Ultrasound/JY/Codes/220919_view_classification/bmode/experiment/model/", help="GPU number configuration")
    parser.add_argument('--Save_Path', type=str, default="/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Model_Save/efficientnet_pretrain_loss_equal/", help="GPU number configuration")

    
    args = parser.parse_args()

    main(args)