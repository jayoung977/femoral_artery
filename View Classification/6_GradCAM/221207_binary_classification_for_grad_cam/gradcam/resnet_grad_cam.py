
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import random
from utils import *
from Lower_Dataset import Lower_Dataset
import natsort
import sys
sys.path.append("/home/wjsrnr20/Works/Project_Vascular/Re_Code_221011/Long_Segmentation/Transfer_Learning/")
from PIL import Image
from Model import resnet50
import target

Class = ['CFA', 'SFA', 'DFA', 'POPA', 'ATA', 'PA', 'PTA', 'DORSALIS']
def main(args):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    torch.set_num_threads(2)

    Root_Files_Path = "/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Data/In_House_Data/Ver1_to_3_integrated/Bmode_png/"
    Root_Class_Folders = natsort.natsorted(os.listdir(Root_Files_Path))

    Root_Class_Paths = []
    for Class_len in range(len(Root_Class_Folders)):
        Class_Path = Root_Files_Path + Root_Class_Folders[Class_len] + "/"
        Root_Class_Paths.append(Class_Path)
    
    print("Class Len")
    print(len(Root_Class_Paths))

    Test_Patient_List = ['P030', 'P031', 'P032', 'P033', 'P035','R001','R002','R003']
    
    print("Test_Patient_List")
    print(len(Test_Patient_List))
    # YI_test_patient = ['Y003','Y004','Y005','Y006']

    Test_Patient_Files, Test_Patient_Class = [], [0 for width in range(len(Root_Class_Paths))]
    for Class_len in range(len(Root_Class_Paths)):
        Class_Path = Root_Class_Paths[Class_len]
        Class_Patient_List = natsort.natsorted(os.listdir(Class_Path))
        Class_Patient_List = List_Cleaning(Class_Patient_List)

        for Patient_len in range(len(Class_Patient_List)):
        # for Patient_len in range(1):
            Patient_Name = Class_Patient_List[Patient_len].split('_')[0]
            Patient_Class = Class_Path.split('/')[-2]

            if(Patient_Name in Test_Patient_List):
                Test_Patient_Files.append(Class_Path + Class_Patient_List[Patient_len])
                Test_Patient_Class[int(Patient_Class)] += 1

    print(len(Test_Patient_Files), np.sum(Test_Patient_Class))

    trnsfm2 = transforms.Compose([transforms.ToTensor()])

    # Target_Class = [25, 51]
    Target_Class = target.target_class[0]
    Test_Class_dset = Lower_Dataset(Test_Patient_Files, Target_Class, None, trnsfm2, Is_Train = 0)
    test_loader = DataLoader(Test_Class_dset, batch_size=args.batch_size, shuffle=True)



    num_classes = 2
    # print("num_classes")
    # print(num_classes)

    model = resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)


    last_model_root_path = "/mnt/NAS01/KH/Project_Results/Vascular_Ultrasound_221205/Classification/Vascular_Ultrasound/Binary_Label/" + str(target.target_class[0]) +'/Model_Save/'
    model_file_max_num = max([int(f_name[-7]) for f_name in os.listdir(last_model_root_path)])
    model_file = [f_name for f_name in os.listdir(last_model_root_path) if str(model_file_max_num) == f_name[-7]][0]

    last_model_saved = last_model_root_path + model_file
    print('last_model_saved: ',last_model_saved)
    model_data = torch.load(last_model_saved, map_location='cpu')
    model.load_state_dict(model_data)

    finalconv_name = model.conv5_x[2].residual_function[6]

    # inference mode

    model.eval()

    # number of result
    num_result = 5


    feature_blobs = []
    backward_feature = []

    # output으로 나오는 feature를 feature_blobs에 append하도록
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())
        
    # Grad-CAM
    def backward_hook(module, input, output):

        backward_feature.append(output[0])

        
    finalconv_name.register_forward_hook(hook_feature)
    finalconv_name.register_backward_hook(backward_hook)

    # get the softmax weight

    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().detach().numpy()) # [2, 512]

    if os.path.isdir(args.Img_Save_Path) != True:
        os.mkdir(args.Img_Save_Path)
    
    Gradcam_Root_Path = args.Img_Save_Path + target.target_name[0]  +'/'
    print('Gradcam_Root_Path: ',Gradcam_Root_Path)
    if os.path.isdir(Gradcam_Root_Path) != True:
        os.mkdir(Gradcam_Root_Path)
    #matched save path     
    matched_Gradcam_Save_Path = Gradcam_Root_Path +'matched/'
    if os.path.isdir(matched_Gradcam_Save_Path) != True:
        os.mkdir(matched_Gradcam_Save_Path)
    classes_name = target.target_class_names
    class_for_path = matched_Gradcam_Save_Path + classes_name[0] +'/'
    if os.path.isdir(class_for_path) != True:
        for i in range(len(classes_name)):
            os.mkdir(matched_Gradcam_Save_Path + classes_name[i])

    # mismatched save path
    mismatched_Gradcam_Save_Path = Gradcam_Root_Path +'mismatched/'
    if os.path.isdir(mismatched_Gradcam_Save_Path) != True:
        os.mkdir(mismatched_Gradcam_Save_Path)
    classes_name = target.target_class_names
    class_for_path = mismatched_Gradcam_Save_Path + classes_name[0] +'/'
    if os.path.isdir(class_for_path) != True:
        for i in range(len(classes_name)):
            os.mkdir(mismatched_Gradcam_Save_Path + classes_name[i])

    # classes_name = target.target_class_names


    
     

    # generate the class activation maps
    import cv2
    def returnCAM(feature_conv, weight_softmax, class_idx):
        size_upsample = (224, 224)
        _, nc, h, w = feature_conv.shape # nc : number of channel, h: height, w: width
        output_cam = []
        # weight 중에서 class index에 해당하는 것만 뽑은 다음, 이를 conv feature와 곱연산
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w))) 
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    from torchvision.utils import make_grid, save_image
    # i = 0
    for Test_Iter, (V_Images, v_Labels, file_path) in enumerate(test_loader):
        
        File_name = file_path[0]
        Y = v_Labels.clone().detach()
        # save_path = Gradcam_Save_Path + classes_name[v_Labels] +'/'


        # 모델의 input으로 주기 위한 image는 따로 설정
        image_for_model = V_Images.clone().detach().cpu()

        # Image denormalization, using mean and std that i was used.

        # 모델의 input으로 사용하도록.
        image_tensor = image_for_model.to(device)
        # logit = ResNet(image_tensor)
        logit = model(image_tensor)
        _, predicted = torch.max(logit, 1)  #가장 큰 값을 갖는 인덱스  ex, 1(cat) 1(cat) 2(car) 3(ship)..... / dim=1 행방향으로 최댓값 찾음 
        predicted = predicted.detach().cpu()



        h_x = F.softmax(logit, dim=1).data.squeeze()

        
        probs, idx = h_x.sort(0, True) #dim = 0, descending = True 


        
        # ============================= #
        # ==== Grad-CAM main lines ==== #
        # ============================= #


        score = logit[:, idx[0]].squeeze() # 예측값 y^c
        score.backward(retain_graph = True) # 예측값 y^c에 대해서 backprop 진행
        
        activations = torch.Tensor(feature_blobs[0]).to(device) # (1, 512, 7, 7), forward activations
        gradients = backward_feature[0] # (1, 512, 7, 7), backward gradients
        b, k, u, v = gradients.size()
        
        alpha = gradients.view(b, k, -1).mean(2) # (1, 512, 7*7) => (1, 512), feature map k의 'importance'
        weights = alpha.view(b, k, 1, 1) # (1, 512, 1, 1)
        
        grad_cam_map = (weights*activations).sum(1, keepdim = True) # alpha * A^k = (1, 512, 7, 7) => (1, 1, 7, 7)
        grad_cam_map = F.relu(grad_cam_map) # Apply R e L U
        grad_cam_map = F.interpolate(grad_cam_map, size=(240, 320), mode='bilinear', align_corners=False) # (1, 1, 224, 224)
        map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
        grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min).data # (1, 1, 224, 224), min-max scaling

   
        # grad_cam_map.squeeze() : (224, 224)
        grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET) # (224, 224, 3), numpy 
        grad_heatmap = torch.from_numpy(grad_heatmap).permute(2, 0, 1).float().div(255) # (3, 224, 224)
        b, g, r = grad_heatmap.split(1)
        grad_heatmap = torch.cat([r, g, b]) # (3, 244, 244), opencv's default format is BGR, so we need to change it as RGB format.

        # save_image(grad_heatmap, os.path.join(save_path, File_name.replace('.png', '_Grad_CAM.png')))
 
        
        grad_result = grad_heatmap + V_Images.cpu() # (1, 3, 224, 224)
        grad_result = grad_result.div(grad_result.max()).squeeze() # (3, 224, 224)
        
        # save_image(grad_result, os.path.join(save_path, File_name.replace('.png', '_Grad_CAM_Img.png')))

        
        
        image_list = []
        
        image_list.append(torch.stack([V_Images.squeeze().cpu(), grad_heatmap, grad_result], 0)) # (3, 3, 224, 224)
        
        images = make_grid(torch.cat(image_list, 0), nrow = 3)
        
        # save_image(images, os.path.join(save_path, File_name.replace('.png', '_Final_Result.png')))
    

        #클래스별 예측 

        if Y == predicted:
            save_path = matched_Gradcam_Save_Path + classes_name[Y]
            if len(os.listdir(save_path)) !=0:
                exist_files = list(set([ path[:9] for path in os.listdir(save_path) if 'png' in path]))
                if File_name[:9] not in exist_files:
                    save_image(images, os.path.join(save_path, File_name.replace('.png', '_Final_Result.png')))
            else:
                save_image(images, os.path.join(save_path, File_name.replace('.png', '_Final_Result.png')))
        else :
            save_path = mismatched_Gradcam_Save_Path + classes_name[Y] + '/'
            if len(os.listdir(save_path)) !=0:
                exist_files =  list(set([ path.split('_')[3]+'_'+path.split('_')[4] for path in os.listdir(save_path) if 'png' in path]))
                if File_name[:9] not in exist_files:
                    save_path_2 = save_path + "Predict_{}_".format(classes_name[predicted]) + File_name.replace('.png', '_Final_Result.png')
                    save_image(images, save_path_2)
            else:
                save_path_2 = save_path + "Predict_{}_".format(classes_name[predicted]) + File_name.replace('.png', '_Final_Result.png')
                save_image(images, save_path_2)
        # i += 1

        # if i  == num_result:
        #     break
            
        feature_blobs.clear()
        backward_feature.clear()

    feature_blobs.clear()
    backward_feature.clear()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', type=str, default="3", help="GPU number configuration")
    parser.add_argument('--batch_size', type=int, default=1, help="GPU number configuration")
    parser.add_argument('--Img_Save_Path', type=str, default="/mnt/NAS01/KH/Project_Results/Vascular_Ultrasound_221205/Classification/Vascular_Ultrasound/Binary_Label/" + str(target.target_class[0]) +'/Gradcam_Save/', help="GPU number configuration")
    
    args = parser.parse_args()

    main(args)