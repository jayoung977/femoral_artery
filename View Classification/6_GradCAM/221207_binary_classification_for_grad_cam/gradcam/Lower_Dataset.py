from PIL import Image
import numpy as np
import torch
import os
import natsort
from torchvision import transforms
import random

def Random_Crop_and_Resizing(image, mask):
    image, mask = np.array(image), np.array(mask)

    Check, Count = 0, 0
    Rand_Crop_Width, Rand_Crop_Hegith = random.randint(100, 200), random.randint(100, 200)
    Rand_Crop_X, Rand_Crop_Y = random.randint(0, np.shape(image)[1] - Rand_Crop_Width), random.randint(0, np.shape(image)[0] - Rand_Crop_Hegith)

    Cropped_Image = image[Rand_Crop_Y : Rand_Crop_Y + Rand_Crop_Hegith, Rand_Crop_X : Rand_Crop_X + Rand_Crop_Width]
    Cropped_Mask = mask[Rand_Crop_Y : Rand_Crop_Y + Rand_Crop_Hegith, Rand_Crop_X : Rand_Crop_X + Rand_Crop_Width]

    Cropped_Image, Cropped_Mask = Image.fromarray(np.uint8(Cropped_Image)), Image.fromarray(np.uint8(Cropped_Mask))
    wanted_width, wanted_height = np.shape(image)[1], np.shape(image)[0]
    Resized_Cropped_Img, Resized_Cropped_Mask = resize_with_padding(Cropped_Image, (wanted_width, wanted_height)), resize_with_padding(Cropped_Mask, (wanted_width, wanted_height))

    Resized_Cropped_Img, Resized_Cropped_Mask= np.array(Resized_Cropped_Img), np.array(Resized_Cropped_Mask)

    X, Y = np.where(Resized_Cropped_Mask[:,:] != 0)

    # if(len(X) != 0):
    if(len(X) >= int(np.shape(Cropped_Image)[1] * np.shape(Cropped_Image)[0] * 0.3)):
        Check = 1

        Count += 1

    return Image.fromarray(np.uint8(Resized_Cropped_Img)), Image.fromarray(np.uint8(Resized_Cropped_Mask))

from PIL import ImageOps
def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0] 
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_and_resize_with_padding(image):
    Check, Count = 0, 0

    image = Image.fromarray(np.uint8(image))
    width, height = np.shape(image)[1], np.shape(image)[0]
    resize_ratio = random.uniform(0.20, 2)

    resized_width, resized_height = int(width * resize_ratio), int(height * resize_ratio)

    resized_image = image.resize((resized_width, resized_height))

    wanted_width, wanted_height = np.shape(image)[1], np.shape(image)[0]
    resized_image = resize_with_padding(resized_image, (wanted_width, wanted_height))

    resized_image = np.array(resized_image)

    # print("resize_and_resize_with_padding")
    # print(np.shape(resized_image))

    return Image.fromarray(np.uint8(resized_image))

import cv2 as cv


def random_area_copy(image, mask, Where_Flip):

    if(Where_Flip == 1): # Lt_Rt_Flip
        Fliped_Img = cv.flip(image, 1)
        Fliped_Mask = cv.flip(mask, 1)
    else: # Up_Dw_Flip
        Fliped_Img = cv.flip(image, 0)
        Fliped_Mask = cv.flip(mask, 0)   

    None_B_X, None_B_Y = np.where(Fliped_Mask[:,:] != 0)
    Min_X, Max_X, Mean_X = np.min(None_B_X), np.max(None_B_X), int(np.mean(None_B_X))
    Min_Y, Max_Y, Mean_Y = np.min(None_B_Y), np.max(None_B_Y), int(np.mean(None_B_Y))

    Random_Area_Img, Random_Area_Mask = image, mask
    if(Where_Flip == 1):
        Fliped_Img = cv.flip(image, 1)
        Fliped_Mask = cv.flip(mask, 1)

        Random_X = random.randint(Min_X, Mean_X)
        Random_Area_Img[:, Random_X:, :] = Fliped_Img[:, Random_X:, :]
        Random_Area_Mask[:, Random_X:] = Fliped_Mask[:, Random_X:]

    else:
        Fliped_Img = cv.flip(image, 0)
        Fliped_Mask = cv.flip(mask, 0) 

        Random_Y = random.randint(Min_Y, Mean_Y)
        Random_Area_Img[Random_Y:, :, :] = Fliped_Img[Random_Y:, :, :]
        Random_Area_Mask[Random_Y:, :] = Fliped_Mask[Random_Y:, :]

    return np.array(Random_Area_Img), np.array(Random_Area_Mask)

def Resize(image, mask):
    image, mask = Image.fromarray(np.uint8(image)), Image.fromarray(np.uint8(mask))
    width, height = np.shape(image)[1], np.shape(image)[0]
    # resize_ratio = random.uniform(0.5, 2)
    resize_ratio = random.uniform(0.3, 0.75)

    # resized_width, resized_height = width, int(height / resize_ratio)
    resized_width, resized_height = width, int(height * resize_ratio) # Set Width Default
    
    if (resized_height > 10):
        resized_image = image.resize((resized_width, resized_height))
        resized_mask = mask.resize((resized_width, resized_height))
        return resized_image, resized_mask
    else:
        return image, mask

def get_concat_v(im1, im2):
    im1, im2 = Image.fromarray(np.uint8(im1)), Image.fromarray(np.uint8(im2)),
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def get_concat_h_ROI(im1, im2, im3):
    im1, im2, im3 = Image.fromarray(np.uint8(im1)), Image.fromarray(np.uint8(im2)), Image.fromarray(np.uint8(im3))
    dst = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.paste(im3, (im1.width + im2.width, 0))
    return dst

def get_concat_v_ROI(im1, im2, im3):
    im1, im2, im3 = Image.fromarray(np.uint8(im1)), Image.fromarray(np.uint8(im2)), Image.fromarray(np.uint8(im3))
    dst = Image.new('RGB', (im1.width, im1.height + im2.height + im3.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    dst.paste(im3, (0, im1.height + im2.height))
    return dst
def Get_X_Y(Coordi_List):
    Coordi_List_Ori = Coordi_List
    ### Leftest_X
    Coordi_List = Coordi_List_Ori
    Coordi_List.sort(key=lambda x : (x[1], x[0]))
    
    if(len(Coordi_List) == 0):
        return 0, 0, 0, 0
    else:
        Leftest_X = Coordi_List[0][1]

        ### Rightest_X
        Coordi_List = Coordi_List_Ori
        Coordi_List.sort(key=lambda x : (-x[1], x[0]))
        Rightest_X = Coordi_List[0][1]

        ### Upeset_Y
        Coordi_List = Coordi_List_Ori
        Coordi_List.sort(key=lambda x : (x[0], x[1]))
        Upeset_Y = Coordi_List[0][0]

        ### Downeset_Y
        Coordi_List = Coordi_List_Ori
        Coordi_List.sort(key=lambda x : (-x[0], x[1]))
        Downeset_Y = Coordi_List[0][0]

        return Leftest_X, Rightest_X, Upeset_Y, Downeset_Y
    
def resize_and_resize_with_Realistic_Padding(Ori_image, mask):
    Ori_image = np.array(Ori_image)
    mask = np.array(mask)

    X, Y = np.where(mask[:,:] == 255)

    ### Leftest_X
    Coordi_List = list(np.array([X, Y]).T)
    Lx, Rx, Uy, Dy = Get_X_Y(Coordi_List)

    ### Get Realistic ROI Regions
    Up_ROI = Ori_image[25 : Uy, 55:320-55, :]
    Center_ROI = Ori_image[Uy : Dy, 55:320-55, :]
    Dw_ROI = Ori_image[Dy : 240-25, 55:320-55, :]

    Up_ROI_GT = mask[25 : Uy, 55:320-55]
    Center_ROI_GT = mask[Uy : Dy, 55:320-55]
    Dw_ROI_GT = mask[Dy : 240-25, 55:320-55]

    if(np.shape(Up_ROI_GT)[0] > 10):
        Resized_Center_ROI, Resized_GT = Resize(Center_ROI, Center_ROI_GT)
        Resized_Center_ROI, Resized_GT = np.array(Resized_Center_ROI), np.array(Resized_GT)

        Up_Dw_Center_Null = np.shape(Center_ROI)[0] - np.shape(Resized_Center_ROI)[0]
        if(Up_Dw_Center_Null % 2 == 0):
            Center_Up_Zeros, Center_Dw_Zeros = int(Up_Dw_Center_Null / 2), int(Up_Dw_Center_Null / 2)
        else:
            Center_Up_Zeros, Center_Dw_Zeros = int(Up_Dw_Center_Null / 2), int(Up_Dw_Center_Null / 2) + 1

        Up_ROI_Pad = np.pad(Up_ROI, pad_width = ((0, Center_Up_Zeros), (0, 0), (0, 0)), mode = 'reflect')
        Dw_ROI_Pad = np.pad(Dw_ROI, pad_width = ((Center_Dw_Zeros, 0), (0, 0), (0, 0)), mode = 'reflect')

        if(np.shape(Resized_Center_ROI)[1] > 10):
            Up_ROI_GT = np.zeros((np.shape(Up_ROI)[0], np.shape(Up_ROI)[1]))
            Dw_ROI_GT = np.zeros((np.shape(Dw_ROI)[0], np.shape(Dw_ROI)[1]))
            Up_ROI_Pad_GT = np.pad(Up_ROI_GT, pad_width = ((0, Center_Up_Zeros), (0, 0)))
            Dw_ROI_Pad_GT = np.pad(Dw_ROI_GT, pad_width = ((Center_Dw_Zeros, 0), (0, 0)))

            Lt_Rt_Center_Null = np.shape(Center_ROI)[1] - np.shape(Resized_Center_ROI)[1]
            if(Lt_Rt_Center_Null % 2 == 0):
                Center_Lt_Zeros, Center_Rt_Zeros = int(Lt_Rt_Center_Null / 2), int(Lt_Rt_Center_Null / 2)
            else:
                Center_Lt_Zeros, Center_Rt_Zeros = int(Lt_Rt_Center_Null / 2), int(Lt_Rt_Center_Null / 2) + 1

            Lt_Up, Rt_Up = Up_ROI[ : , : Center_Lt_Zeros, :], Up_ROI[ : , -Center_Rt_Zeros : , :]
            Lt_Dw, Rt_Dw = Dw_ROI[ : , : Center_Lt_Zeros, :], Dw_ROI[ : , -Center_Rt_Zeros : , :]

            Lt_Up_Pad, Rt_Up_Pad = np.pad(Lt_Up, pad_width = ((0, int(np.shape(Resized_Center_ROI)[0] / 2)), (0, 0), (0, 0)), mode = 'reflect'), np.pad(Rt_Up, pad_width = ((0, int(np.shape(Resized_Center_ROI)[0] / 2)), (0, 0), (0, 0)), mode = 'reflect')
            Lt_Dw_Pad, Rt_Dw_Pad = np.pad(Lt_Dw, pad_width = ((int(np.shape(Resized_Center_ROI)[0] / 2), 0), (0, 0), (0, 0)), mode = 'reflect'), np.pad(Rt_Dw, pad_width = ((int(np.shape(Resized_Center_ROI)[0] / 2), 0), (0, 0), (0, 0)), mode = 'reflect')

            Lt_Pad = np.array(get_concat_v(Lt_Up_Pad, Lt_Dw_Pad))
            Rt_Pad = np.array(get_concat_v(Rt_Up_Pad, Rt_Dw_Pad))

            #############
            Lt_Up_GT, Rt_Up_GT = Up_ROI_GT[ : , : Center_Lt_Zeros], Up_ROI_GT[ : , -Center_Rt_Zeros :]
            Lt_Dw_GT, Rt_Dw_GT = Dw_ROI_GT[ : , : Center_Lt_Zeros], Dw_ROI_GT[ : , -Center_Rt_Zeros :]

            Lt_Up_Pad_GT, Rt_Up_Pad_GT = np.pad(Lt_Up_GT, pad_width = ((0, int(np.shape(Resized_Center_ROI)[0] / 2)), (0, 0))), np.pad(Rt_Up_GT, pad_width = ((0, int(np.shape(Resized_Center_ROI)[0] / 2)), (0, 0)))
            Lt_Dw_Pad_GT, Rt_Dw_Pad_GT = np.pad(Lt_Dw_GT, pad_width = ((int(np.shape(Resized_Center_ROI)[0] / 2), 0), (0, 0))), np.pad(Rt_Dw_GT, pad_width = ((int(np.shape(Resized_Center_ROI)[0] / 2), 0), (0, 0)))

            Lt_Pad_GT = np.array(get_concat_v(Lt_Up_Pad_GT, Lt_Dw_Pad_GT))
            Rt_Pad_GT = np.array(get_concat_v(Rt_Up_Pad_GT, Rt_Dw_Pad_GT))



            # Lt_Pad_Idx = int((np.shape(Lt_Pad)[0] - Up_Dw_Center_Null))
            Lt_Pad_Idx = int((np.shape(Up_ROI)[0] / 2))

            Lt_Pad = Lt_Pad[Lt_Pad_Idx : Lt_Pad_Idx + np.shape(Resized_Center_ROI)[0], :, :]
            Rt_Pad = Rt_Pad[Lt_Pad_Idx : Lt_Pad_Idx + np.shape(Resized_Center_ROI)[0], :, :]

            Resized_Center_ROI = get_concat_h_ROI(Lt_Pad, Resized_Center_ROI, Rt_Pad)

            ############################
            Lt_Pad_GT = Lt_Pad_GT[Lt_Pad_Idx : Lt_Pad_Idx + np.shape(Resized_Center_ROI)[0], :, :]
            Rt_Pad_GT = Rt_Pad_GT[Lt_Pad_Idx : Lt_Pad_Idx + np.shape(Resized_Center_ROI)[0], :, :]

            Resized_GT = get_concat_h_ROI(Lt_Pad_GT, Resized_GT, Rt_Pad_GT)

            Final = get_concat_v_ROI(Up_ROI_Pad, Resized_Center_ROI, Dw_ROI_Pad)
            Final_GT = get_concat_v_ROI(Up_ROI_Pad_GT, Resized_GT, Dw_ROI_Pad_GT)
            
            Final = resize_with_padding(Final, (np.shape(Ori_image)[1], np.shape(Ori_image)[0]))
            Final_GT = resize_with_padding(Final_GT, (np.shape(Ori_image)[1], np.shape(Ori_image)[0])).convert('L')
            
            return Final, Final_GT
        else:
            return Ori_image, mask
    else:
        return Ori_image, mask

def Resize_Height(image, mask, Is_Center = 0):
    Check, Count = 0, 0

    image, mask = Image.fromarray(np.uint8(image)), Image.fromarray(np.uint8(mask))
    width, height = np.shape(image)[1], np.shape(image)[0]
    # resize_ratio = random.uniform(0.5, 2)

    if(Is_Center == 0):
        resize_ratio = random.uniform(1.5, 3)
    elif(Is_Center == 1):
        resize_ratio = random.uniform(0.3, 0.5)
    else:
        resize_ratio = random.uniform(0.5, 1)

    resized_width, resized_height = width, int(height * resize_ratio) # Set Width Default
    # resized_image = image.resize((resized_width, resized_height))
    # resized_mask = mask.resize((resized_width, resized_height))
    # return resized_image, resized_mask

    if (resized_height > 10):
        resized_image = image.resize((resized_width, resized_height))
        resized_mask = mask.resize((resized_width, resized_height))
        return resized_image, resized_mask
    else:
        return image, mask


def Make_Zoomed_View(img, mask):
    Ori_image, mask = np.array(img), np.array(mask)

    X, Y = np.where(mask[:,:] == 255)

    ### Leftest_X
    Coordi_List = list(np.array([X, Y]).T)
    Lx, Rx, Uy, Dy = Get_X_Y(Coordi_List)

    if(Uy != 0 and Dy != 0):
        ### Get Realistic ROI Regions
        Up_ROI = Ori_image[25 : Uy, 55:320-55, :]
        Center_ROI = Ori_image[Uy : Dy, 55:320-55, :]
        Dw_ROI = Ori_image[Dy : 240-25, 55:320-55, :]

        Up_ROI_GT = mask[25 : Uy, 55:320-55]
        Center_ROI_GT = mask[Uy : Dy, 55:320-55]
        Dw_ROI_GT = mask[Dy : 240-25, 55:320-55]

        if((np.shape(Up_ROI)[0] != 0 and np.shape(Center_ROI)[0] != 0 and np.shape(Dw_ROI)[0] != 0)):
            Up_ROI, Up_ROI_GT = Resize_Height(Up_ROI, Up_ROI_GT, Is_Center = 0)
            Center_ROI, Center_ROI_GT = Resize_Height(Center_ROI, Center_ROI_GT, Is_Center = 1)

            Dw_Resize = random.randint(0, 5)
            if(Dw_Resize == 0):
                Dw_ROI, Dw_ROI_GT = Resize_Height(Dw_ROI, Dw_ROI_GT, Is_Center = 2)
            else:
                Dw_ROI, Dw_ROI_GT = Resize_Height(Dw_ROI, Dw_ROI_GT, Is_Center = 0)

            Final = get_concat_v_ROI(Up_ROI, Center_ROI, Dw_ROI)
            Final_GT = get_concat_v_ROI(Up_ROI_GT, Center_ROI_GT, Dw_ROI_GT)

            Final = resize_with_padding(Final, (np.shape(Ori_image)[1], np.shape(Ori_image)[0]))
            Final_GT = resize_with_padding(Final_GT, (np.shape(Ori_image)[1], np.shape(Ori_image)[0])).convert('L')

            return Final, Final_GT
        else:
            return Image.fromarray(np.uint8(Ori_image)), Image.fromarray(np.uint8(mask))
    else:
        return Image.fromarray(np.uint8(Ori_image)), Image.fromarray(np.uint8(mask))

# [Gaussian_Blur, Box_Blur, Unsharp_Blur, Rank_Blur, Median_Blur, Min_Blur, Max_Blur, Model_Blur]

import PIL
def Flip_Horizon(img, mask):  # not from the paper
    return PIL.ImageOps.mirror(img), PIL.ImageOps.mirror(mask)

def Flip_Vertical(img, mask):  # not from the paper
    return PIL.ImageOps.flip(img), PIL.ImageOps.flip(mask)

def ShearX(image, v):  # [-0.3, 0.3]
    image = Image.fromarray(np.uint8(image))
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return image.transform(image.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(image, v):  # [-0.3, 0.3]
    image = Image.fromarray(np.uint8(image))
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return image.transform(image.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert 1 <= v <= 60
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)

# RIGHT_DORSALIS_LONG LEFT_DORSALIS_LONG
# 25 51
class Lower_Dataset():
    def __init__(self, Files_set, Target_Class, transform, transform2, Is_Train):
        self.transform = transform
        self.transform2 = transform2

        self.Img_file_list = Files_set
        
        self.Label_True_Img_list = []
        self.Label_False_Img_list = []
        
        self.Label_True_List = []
        self.Label_False_List = []

        self.Is_Train = Is_Train

        self.Target_Class = Target_Class
        print("self.Target_Class")
        print(self.Target_Class)

        self.Binary_Class = [0 for width in range(len(self.Target_Class))]
        for F_len in range(len(self.Img_file_list)):
            File_Class = int((self.Img_file_list[F_len].split('/'))[-2])
            # 25 51
            
            if(File_Class in self.Target_Class):
                self.Label_True_Img_list.append(self.Img_file_list[F_len])
                self.Label_True_List.append(torch.tensor(int(0)))
                # print(File_Class, self.Img_file_list[F_len])
                self.Binary_Class[0] += 1
            else:
                self.Label_False_Img_list.append(self.Img_file_list[F_len])
                self.Label_False_List.append(torch.tensor(int(1)))

                self.Binary_Class[1] += 1

        print("##############")
        print(len(self.Label_True_Img_list), len(self.Label_True_List))
        print(len(self.Label_False_Img_list), len(self.Label_False_List))
        print(self.Binary_Class)
        print("##############")

        # Shuffle_Target List and Balancing
        Shuffle_Idx = np.arange(len(self.Label_True_Img_list))
        np.random.shuffle(Shuffle_Idx)
        self.Label_True_Img_list, self.Label_True_List = np.array(self.Label_True_Img_list), np.array(self.Label_True_List)
        self.Label_True_Img_list, self.Label_True_List = self.Label_True_Img_list[Shuffle_Idx], self.Label_True_List[Shuffle_Idx]

        # Shuffle None_Target List and Balancing
        Shuffle_Idx = np.arange(len(self.Label_False_Img_list))
        np.random.shuffle(Shuffle_Idx)
        self.Label_False_Img_list, self.Label_False_List = np.array(self.Label_False_Img_list), np.array(self.Label_False_List)
        self.Label_False_Img_list, self.Label_False_List = self.Label_False_Img_list[Shuffle_Idx], self.Label_False_List[Shuffle_Idx]
        self.Label_False_Img_list, self.Label_False_List = self.Label_False_Img_list[:int(len(self.Label_True_Img_list) * 5)], self.Label_False_List[:int(len(self.Label_True_Img_list) * 5)]

        self.Label_True_Img_list, self.Label_False_Img_list = list(self.Label_True_Img_list), list(self.Label_False_Img_list)
        self.Label_True_List, self.Label_False_List = list(self.Label_True_List), list(self.Label_False_List)

        self.Total_Img_List = self.Label_True_Img_list + self.Label_False_Img_list
        self.Total_Label_List = self.Label_True_List + self.Label_False_List
        
        if(self.Is_Train == 1):
            Shuffle_Idx = np.arange(len(self.Total_Img_List))
            np.random.shuffle(Shuffle_Idx)   
            self.Total_Img_List, self.Total_Label_List = np.array(self.Total_Img_List), np.array(self.Total_Label_List)
            self.Total_Img_List, self.Total_Label_List = self.Total_Img_List[Shuffle_Idx], self.Total_Label_List[Shuffle_Idx]

        self.Label_True_Img_list, self.Label_False_Img_list = list(self.Label_True_Img_list), list(self.Label_False_Img_list)
        self.Label_True_List, self.Label_False_List = list(self.Label_True_List), list(self.Label_False_List)

        self.Total_Img_List = self.Label_True_Img_list + self.Label_False_Img_list
        self.Total_Label_List = self.Label_True_List + self.Label_False_List

        # self.Binary_Class = [0 for width in range(len(self.Target_Class))]
        # for F_len in range(len(self.Img_file_list)):
        #     File_Class = int((self.Img_file_list[F_len].split('/'))[-2])
        #     # 25 51
            
        #     if(File_Class in self.Target_Class):
        #         self.Label_list.append(torch.tensor(int(0)))
        #         self.Binary_Class[0] += 1
        #     else:
        #         self.Label_list.append(torch.tensor(int(1)))
        #         self.Binary_Class[1] += 1

        #     # self.Label_file_list.append(torch.tensor(int(File_Class)))
        # print(self.Binary_Class)

        if(self.Is_Train == 1):
            print("Train")
            print("True Img, False Img")
            print(len(self.Label_True_Img_list), len(self.Label_False_Img_list))
            print("Total Img, Total Label")
            print(len(self.Total_Img_List), len(self.Total_Label_List))
        
        else:
            print("Test")
            print("True Img, False Img")
            print(len(self.Label_True_Img_list), len(self.Label_False_Img_list))
            print("Total Img, Total Label")
            print(len(self.Total_Img_List), len(self.Total_Label_List))

    def __len__(self):
        return len(self.Total_Img_List)

    def __getitem__(self, index):
        # self.Total_Img_List self.Total_Label_List
        Ori_img_path, Class_Label = self.Total_Img_List[index], self.Total_Label_List[index]
        Ori_Img = Image.open(Ori_img_path).convert('RGB')
        Ori_Img = Ori_Img.resize((320, 240))

        if (self.Is_Train == True):
            Is_Multi_RandAug = random.randint(0, 5)
            if(Is_Multi_RandAug == 0):
                Is_Horizontal_Aug = random.randint(0,1)
                # Is_Horizontal_Aug = 1
                if(Is_Horizontal_Aug == 0):
                    Ori_Img = np.array(Ori_Img)

                    Div_num = random.randint(2, 5)
                    Width_range = ((320-55) - 55)

                    Div_Width = int(Width_range / Div_num)
                    dst = Image.new('RGB', (320, 240))

                    Concat_Width = 0
                    for D_len in range(Div_num):
                        globals()["Div_{}".format(D_len)] = Ori_Img[ : , 55 + (D_len * Div_Width) : 55 + ((D_len + 1) * Div_Width), :]
                        globals()["Div_{}".format(D_len)] = self.transform(Image.fromarray(np.uint8(globals()["Div_{}".format(D_len)])))
                    
                        Concat_Width += globals()["Div_{}".format(D_len)].width

                        dst.paste(globals()["Div_{}".format(D_len)], (Concat_Width, 0))
                    
                    Ori_Img = dst
                    # print("HorizontalAug")
                    # print(np.shape(Ori_Img))
                else:
                    Ori_Img = np.array(Ori_Img)

                    Div_num = random.randint(2, 5)
                    Width_range = ((240-25) - 25)

                    Div_Width = int(Width_range / Div_num)
                    dst = Image.new('RGB', (320, 240))

                    Concat_Width = 0
                    for D_len in range(Div_num):
                        globals()["Div_{}".format(D_len)] = Ori_Img[25 + (D_len * Div_Width) : 25 + ((D_len + 1) * Div_Width), : , :]
                        globals()["Div_{}".format(D_len)] = self.transform(Image.fromarray(np.uint8(globals()["Div_{}".format(D_len)])))
                
                        Concat_Width += globals()["Div_{}".format(D_len)].height

                        dst.paste(globals()["Div_{}".format(D_len)], (0, Concat_Width))
                    
                    Ori_Img = dst
            else:
                Ori_Img = self.transform(Ori_Img)
                
            ###
            Is_Resize_and_Resize_with_Padding = random.randint(0, 10)
            Is_ShearX, Is_ShearY = random.randint(0, 10), random.randint(0, 10)
            Is_TranslateXabs, Is_TranslateYabs = random.randint(0, 10), random.randint(0, 10)
            Is_Rotate = random.randint(0, 10)
            

            if(Is_Resize_and_Resize_with_Padding == 0):
                Ori_Img = resize_and_resize_with_padding(Ori_Img)
            if(Is_ShearX == 0):
                ShearX_v = random.uniform(-0.3, 0.3)
                Ori_Img = ShearX(Ori_Img, ShearX_v)
            if(Is_ShearY == 0):
                ShearY_v = random.uniform(-0.3, 0.3)
                Ori_Img = ShearY(Ori_Img, ShearY_v)
            if(Is_TranslateXabs == 0):
                TranslateXabs_v = random.randint(10, 30)
                Ori_Img = TranslateXabs(Ori_Img, TranslateXabs_v)
            if(Is_TranslateYabs == 0):
                TranslateYabs_v = random.randint(10, 30)
                Ori_Img = TranslateYabs(Ori_Img, TranslateYabs_v)
            if(Is_Rotate == 0):
                Rotate_v = random.randint(1, 30)
                Ori_Img = Rotate(Ori_Img, Rotate_v)
        else:
            Ori_Img = Ori_Img
            
        Ori_Img = np.array(Ori_Img.resize((320, 240)))
        Ori_Img = Ori_Img.astype('float32')
        Ori_Img = (Ori_Img / 255.0)

        Ori_Img = np.array(Ori_Img)
        # print("Ori_Img Final")
        # print(np.shape(Ori_Img))
        Ori_Img = self.transform2(Ori_Img)

        return Ori_Img, Class_Label, (self.Total_Img_List[index].split('/')[-1])
