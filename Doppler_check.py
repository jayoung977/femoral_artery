#%%
import math
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool,JoinableQueue
from PIL import Image
import os
import time
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool,JoinableQueue
#%%

Root_Path =  "C:/Users/user/Desktop/femoral_artery_with_KH/Results/CCA_220204_3/"
Root_Save_Path = "C:/Users/user/Desktop/femoral_artery_with_KH/Results/CCA_220204_3/"

folder = os.listdir(Root_Path)

sub_folder_path = [ os.path.join(Root_Path,x) for x in folder]
sub_folder_p = [os.listdir(x) for x in sub_folder_path ]
sub_folder_p #0:both, 1:left, 2: right

#0:both_left
both_left_path_0 = [sub_folder_path[0]+'/'+str(x)+'/doppler/left/' for x in sub_folder_p[0]]
both_left_path = [ os.path.join(x,y)for x in both_left_path_0 for y in os.listdir(x)]
both_left_path =[ x for x in both_left_path if '.png' in x]

#0:both_right
both_right_path_0 = [sub_folder_path[0]+'/'+str(x)+'/doppler/right/' for x in sub_folder_p[0]]
both_right_path = [ os.path.join(x,y)for x in both_right_path_0 for y in os.listdir(x)]
both_right_path =[ x for x in both_right_path if '.png' in x]
#1:left
left_path_0 = [sub_folder_path[1]+'/'+str(x)+'/doppler/' for x in sub_folder_p[1]]
left_path = [ os.path.join(x,y)for x in left_path_0 for y in os.listdir(x)]
left_path =[ x for x in left_path if '.png' in x]
#2: right
right_path_0 = [sub_folder_path[2]+'/'+str(x)+'/doppler/' for x in sub_folder_p[2]]
right_path = [ os.path.join(x,y)for x in right_path_0 for y in os.listdir(x)]
right_path =[ x for x in right_path if '.png' in x]

all_data = both_left_path+both_right_path+left_path+right_path


# for i in range(len(sub_folder_p)):
#     if i == 0:
#         for d in sub_folder_p[0]:
#             d = Root_Save_Path + 'both/' + d 
#             d1 = d +'/color/'
#             d2 = d +'/b_mode/'
#             if not os.path.isdir(d1):
#                 os.mkdir(d1)
#                 d1_1 = d1 +'/left/'
#                 d1_2 = d1 +'/right/'
#                 if not os.path.isdir(d1_1) or not os.path.isdir(d1_2):
#                     os.mkdir(d1_1)
#                     os.mkdir(d1_2)
#             elif not os.path.isdir(d2):
#                 os.mkdir(d2)
#                 d2_1 = d2 +'/left/'
#                 d2_2 = d2 +'/right/'
#                 if not os.path.isdir(d2_1) or not os.path.isdir(d2_2):
#                     os.mkdir(d2_1)
#                     os.mkdir(d2_2)
                        
#     elif i == 1:
#         for d in sub_folder_p[1]:
#             d = Root_Save_Path + 'left/' + d
#             d1 = d +'/color/'
#             d2 = d +'/b_mode/'
#             if not os.path.isdir(d1):
#                 os.mkdir(d1)
#             elif not os.path.isdir(d2):
#                 os.mkdir(d2)
                    
#     elif i == 2:                    
#         for d in sub_folder_p[2]:
#             d = Root_Save_Path + 'right/' + d
#             d1 = d +'/color/'
#             d2 = d +'/b_mode/'
#             if not os.path.isdir(d1):
#                 os.mkdir(d1)
#             elif not os.path.isdir(d2):
#                 os.mkdir(d2)


#%%
# all_data[0].replace('doppler','color')
#%%
def Save_File(data):
    for i in range(len(data)):
        img_path = data[i]
        img = Image.open(img_path)
        img.convert("RGB")
        img = np.array(img)
        
        Y,X = np.where((img[:,:,0] > 100)&(img[:,:,1] <50) | (img[:,:,2] > 100)&(img[:,:,1] <50))
        # print(Y,X )
        if (Y.size > 0) & (X.size > 0):
            color_path = img_path.replace('doppler','color')
            img =  Image.fromarray(np.uint8(img))
            img.save(color_path)
            print('[', i,'/',len(data),']',img_path )
            
            
        else:
            b_mode_path = img_path.replace('doppler','b_mode')
            img =  Image.fromarray(np.uint8(img))
            img.save(b_mode_path)
            print( '[', i,'/',len(data),']',img_path )


if __name__ == '__main__':
    with Pool() as p:
        p.map(Save_File, all_data)
        p.close()
        p.join()
       



        # print(img_path)
        # plt.subplot(1,2,1)
        # plt.imshow(img)
        # plt.subplot(1,2,2)
        # plt.imshow(img)
        # plt.scatter(X, Y,c='green',alpha=0.7)
        # plt.figure(figsize=(10,8))
        # plt.show()
#%%
####trial####
# test_data_path = 'C:/Users/user/Desktop/femoral_artery_with_KH/20170322(6)_22.png'


# img = Image.open(test_data_path)
# img.convert("RGB")

# r, g, b = img.split()
# img = np.array(img)
# Y,X = np.where((img[:,:,0] > 200) & (img[:,:,1] > 10)& (img[:,:,2] > 10))

# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.subplot(1,2,2)
# plt.imshow(img)
# plt.scatter(X, Y,c='green',alpha=0.7)
# plt.figure(figsize=())
# plt.show()

