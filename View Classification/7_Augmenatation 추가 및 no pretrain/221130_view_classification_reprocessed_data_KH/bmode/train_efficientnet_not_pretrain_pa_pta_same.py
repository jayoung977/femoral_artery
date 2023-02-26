from cmath import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

import argparse
import os
import Dataset_inhouse_sev
import random
import target_inhouse_sev
from Augmentation_utils import RandAugment


import random
import torch
import numpy as np
import random

random_seed =42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


def main(args):
    print('gpu_num: ',args.gpu_num)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    torch.set_num_threads(1)
    print('-------<target>-------')
    print(target_inhouse_sev.target_name)
    print('num_class: ', len(target_inhouse_sev.target_class))
    print('----------------------')
    ### Train, test ###

    root_path =  "/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Data/In_House_Data/Ver1_to_3_integrated/Bmode_png/"
    all_paths = [ root_path + class_fol +'/'+ path for class_fol in os.listdir(root_path) if 'Thumbs.db' not in class_fol for path in os.listdir(root_path + class_fol) if 'png' in path] 

    test_root_path = "/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Data/Severance_Data/Ver3_to_4_integrated/Vessel/"
    test_all_paths = [ test_root_path + class_fol +'/'+ path for class_fol in os.listdir(test_root_path) if 'Thumbs.db' not in class_fol for path in os.listdir(test_root_path + class_fol) if 'png' in path ] 
    # print('test_all_paths_sample:',test_all_paths[0])

    random.shuffle(all_paths)

    trnsfm = transforms.Compose([])
    trnsfm.transforms.insert(0, RandAugment(args.Aug, 23))
    trnsfm2 = transforms.Compose([transforms.ToTensor()])


    train_dataset = Dataset_inhouse_sev.Dataset(all_paths, trnsfm, trnsfm2, True)
    test_dataset = Dataset_inhouse_sev.Dataset(test_all_paths,trnsfm, trnsfm2, False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    num_classes = len(target_inhouse_sev.target_class)


    model = models.efficientnet_b3(pretrained=False)



    # print(model)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1]= torch.nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


    Model_Save_Path = args.Save_Path  + target_inhouse_sev.target_name +'/'
    print('Model_Save_Path: ',Model_Save_Path)
    if os.path.isdir(args.Save_Path) != True:
        os.mkdir(args.Save_Path)
    if os.path.isdir(Model_Save_Path) != True:
        os.mkdir(Model_Save_Path)

    weight_files = [ Model_Save_Path + file_name for file_name in os.listdir(Model_Save_Path)]
    weight_files_epoch = [ int(file_name.split('Epoch_')[-1].split('_')[0]) for file_name in os.listdir(Model_Save_Path) ]


    if (len(weight_files)!= 0):
        max_epoch_index = weight_files_epoch.index(max(weight_files_epoch))
        last_model_saved = weight_files[max_epoch_index]
        model_data = torch.load(last_model_saved)
        model.load_state_dict(model_data['model_state_dict'])
        optimizer.load_state_dict(model_data['optimizer_state_dict'])


        start_epoch = model_data['epoch'] + 1
        print('start_epoch: ',start_epoch)
        
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

        train_loss_mean = train_loss_sum / train_total_batch
        train_acc_mean = train_acc_sum / train_total
        print("Training Epoch : %d, Cross_Entropy_Loss : %.5f, Accuracy : %.5f" % (epoch , train_loss_mean, train_acc_mean))
        
        #클래스별 예측 
        classes = target_inhouse_sev.target_class_names
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
                

            } , os.path.join(Model_Save_Path + "Best_ACC_%.5f at Epoch_%d_Model.tar" % (best_accuracy, epoch)))


        else:
            if (bf_test_accuracy_epoch != 0) and (bf_test_accuracy != 0) :
                os.remove(Model_Save_Path + "Best_NOT_ACC_%.5f at Epoch_%d_Model.tar" % (bf_test_accuracy, bf_test_accuracy_epoch))
            bf_test_accuracy_epoch = epoch
            bf_test_accuracy = test_acc_mean 
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch' : epoch,
                

            } , os.path.join(Model_Save_Path + "Best_NOT_ACC_%.5f at Epoch_%d_Model.tar" % (bf_test_accuracy, bf_test_accuracy_epoch)))
             
                    
        print("-"*100)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', type=str, default="3", help="GPU number configuration")
    parser.add_argument('--batch_size', type=int, default=128, help="GPU number configuration")

    parser.add_argument('--Learning_rate', type=float, default="0.001", help="GPU number configuration")
    parser.add_argument('--Epoch', type=int, default=1000, help="GPU number configuration")

    Aug = 5
    parser.add_argument('--Aug', type=int, default=Aug, help="GPU number configuration")

    # parser.add_argument('--Save_Path', type=str, default="/home/hankh/Works/Project/Vascular_Ultrasound/JY/Codes/220919_view_classification/bmode/experiment/model/", help="GPU number configuration")
    parser.add_argument('--Save_Path', type=str, default="/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Model_Save/Ver1_to_3_integrated/train_efficientnet_not_pretrain_pa_pta_same_only_contrast_Aug_" + str(Aug) + "_Y003_to_Y006/", help="GPU number configuration")

    
    args = parser.parse_args()

    main(args)