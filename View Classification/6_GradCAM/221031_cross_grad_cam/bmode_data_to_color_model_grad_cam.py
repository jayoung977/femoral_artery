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


from Dataset import Dataset


import natsort
from torchsummary import summary

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import random

from torch.autograd import Function
import target 
import torchvision.models as models

class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        # input image 기준으로 양수인 부분만 1로 만드는 positive_mask 생성
        positive_mask = (input_img > 0).type_as(input_img)
        
        # torch.addcmul(input, tensor1, tensor2) => output = input + tensor1 x tensor 2
        # input image와 동일한 사이즈의 torch.zeros를 만든 뒤, input image와 positive_mask를 곱해서 output 생성
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        
        # backward에서 사용될 forward의 input이나 output을 저장
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        
        # forward에서 저장된 saved tensor를 불러오기
        input_img, output = self.saved_tensors
        grad_input = None

        # input image 기준으로 양수인 부분만 1로 만드는 positive_mask 생성
        positive_mask_1 = (input_img > 0).type_as(grad_output)
        
        # 모델의 결과가 양수인 부분만 1로 만드는 positive_mask 생성
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        
        # 먼저 모델의 결과와 positive_mask_1과 곱해주고,
        # 다음으로는 positive_mask_2와 곱해줘서 
        # 모델의 결과가 양수이면서 input image가 양수인 부분만 남도록 만들어줌
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        # 모델이 예측한 결과값을 기준으로 backward 진행
        one_hot.backward(retain_graph=True)

        # input image의 gradient를 저장
        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        output = output.transpose((1, 2, 0))
        return output

def main(args):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    torch.set_num_threads(2)
    print('-------<target>-------')
    print(target.target_name)
    ### Train, test ###

    #bmode
    # root_path = "/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Vessel/"

    #color 
    root_path ="/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Vessel/"




    all_paths = [ root_path + class_fol +'/'+ path for class_fol in os.listdir(root_path) for path in os.listdir(root_path + class_fol) if '.png' in path] 
    random.seed(42)
    random.shuffle(all_paths)

    trnsfm = transforms.Compose([transforms.ToTensor(), transforms.Resize((240, 320))])


    test_dataset = Dataset(all_paths, trnsfm, False)


    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    num_classes = len(target.target_class)


    num_classes = len(target.target_class)
    model = models.efficientnet_b3(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1]= torch.nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
    model.to(device)


    last_model_root_path = "/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Model_Save/color_efficientnet_pretrain/" + target.target_name +'/'
    model_file = os.listdir(last_model_root_path)[0]
    last_model_saved = last_model_root_path + model_file
    print('last_model_saved: ',last_model_saved)
    model_data = torch.load(last_model_saved, map_location='cpu')
    model.load_state_dict(model_data['model_state_dict'])

    finalconv_name = model.features[8][0]

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
    
    Gradcam_Root_Path = args.Img_Save_Path + target.target_name  +'/'
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
    parser.add_argument('--gpu_num', type=str, default="0", help="GPU number configuration")
    parser.add_argument('--batch_size', type=int, default=1, help="GPU number configuration")

    parser.add_argument('--dim_x', type=int, default=320, help="GPU number configuration")
    parser.add_argument('--dim_y', type=int, default=240, help="GPU number configuration")

    parser.add_argument('--Learning_rate', type=float, default="0.001", help="GPU number configuration")
    parser.add_argument('--Epoch', type=int, default=200, help="GPU number configuration")

    parser.add_argument('--Img_Save_Path', type=str, default="/mnt/NAS01/Jayoung/femoral_artery/data2/View_Classification/Gradcam_Save/bmode_data_to_color_efficientnet_pretrain/", help="GPU number configuration")
    

    parser.add_argument('--epsilon', type=float, default=0.00000001)
    
    args = parser.parse_args()

    main(args)
