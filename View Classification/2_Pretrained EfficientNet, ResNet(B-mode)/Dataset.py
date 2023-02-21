from PIL import Image
import numpy as np
import torch
import os
import natsort
import target 
import torch.nn.functional as F
class Dataset():
    def __init__(self, path_list, transform, istrain):
        self.transform = transform
        self.Img_file_list = []
        self.Img_label_list = []
        
        target_class = np.array(target.target_class)
        

        if istrain == True:    #train
            patient = target.train_patient
            print('----train----')
            print(patient)
        else:                  #test
            patient = target.test_patient
            print('----test----')
            print(patient)

        for i in range(len(path_list)):
            patient_num = path_list[i].split('/')[-1][:4]  #P0XX            
            before_class_num = int(path_list[i].split('/')[-2])
            
            if(str(before_class_num) in target_class):                
                after_class_num, _ = np.where(target_class[:,:] == str(before_class_num))  #새로 클래스 정의 
                after_class_num = after_class_num[0] # ex) [1] -> 1
                
                if patient_num in patient:
                    self.Img_file_list.append(path_list[i])
                    self.Img_label_list.append(torch.tensor(int(after_class_num)))
    
        if(istrain == True):
            print("Train Set Shape")
            print(len(self.Img_file_list), len(self.Img_label_list))
        else:
            print("Test Set Shape")
            print(len(self.Img_file_list), len(self.Img_label_list))
        
        class_unique, class_per_num = np.unique(self.Img_label_list,return_counts = True  )
        
        for uni, num in zip(class_unique, class_per_num):
            print(uni,':',num)
        self.class_unique = class_unique
        self.class_per_num = class_per_num

    def value_counts(self):
        return list(self.class_unique),list(self.class_per_num),list(self.Img_label_list)


    def __len__(self):
        return len(self.Img_file_list)

    def __getitem__(self, index):
        Ori_img_path = self.Img_file_list[index] # 데이터셋에서 파일 하나를 특정
        Ori_Img = Image.open(Ori_img_path)

        Ori_Img = np.array(Ori_Img)
        Ori_Img = Ori_Img.astype('float32')
        Ori_Img = (Ori_Img / 255.0)

        Ori_Img_transformed = self.transform(Ori_Img)

        Img_Label = self.Img_label_list[index]
        # Img_Label = F.one_hot(Img_Label, num_classes=2)

        # return Ori_Img_transformed, Img_Label, (self.Img_file_list[index].split('/')[-1])
        return Ori_Img_transformed, Img_Label, (self.Img_file_list[index])
