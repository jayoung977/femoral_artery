#%%
import math
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool,JoinableQueue
from PIL import Image
import os
import time

# dim = 512
Root_Path = "C:/Users/user/Desktop/femoral artery/2022.01.21_KH/"
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

#%%

all_data = both_left_path+both_right_path+left_path+right_path

#%%

for i in range(len(sub_folder_p)):
    if i == 0:
        for d in sub_folder_p[0]:
            d = Root_Save_Path + 'both/' + d 
            if not os.path.isdir(d):
                    os.mkdir(d)
                    d = d +'/doppler/'
                    if not os.path.isdir(d):
                        os.mkdir(d)
                        d1 = d +'/left/'
                        d2 = d +'/right/'
                        if not os.path.isdir(d1) or not os.path.isdir(d2):
                            os.mkdir(d1)
                            os.mkdir(d2)

    if i == 1:
        for d in sub_folder_p[1]:
            d = Root_Save_Path + 'left/' + d
            if not os.path.isdir(d):
                    os.mkdir(d)
                    d = d +'/doppler/'
                    if not os.path.isdir(d):
                        os.mkdir(d)
                    
                        
        for d in sub_folder_p[2]:
            d = Root_Save_Path + 'right/' + d
            if not os.path.isdir(d):
                    os.mkdir(d)
                    d = d +'/doppler/'
                    if not os.path.isdir(d):
                        os.mkdir(d)
                      

temp = 0
def Basic_CCA2(Img, seed_x, seed_y, Visit): # Make Largest Component
    dim_x, dim_y = np.shape(Img)[1], np.shape(Img)[0]

    Q = JoinableQueue() #queue
    L=[]

    count = 0

    Q.put((seed_x,seed_y))
    Visit[seed_y][seed_x] = 1
    L.append((seed_x, seed_y))

    while (Q.qsize() != 0) :#queue is not empty
    # dequeue a point as a centered point

        x, y = Q.get() #centered point , seed

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px = x + dx
                py = y + dy

                # not visited and the value is positive
                if ( (px >= 0) and (px < dim_x) and (py >= 0) and (py < dim_y) and (Visit[py][px] == 0) ): 
                    R, G, B = Img.getpixel((px,py))[0], Img.getpixel((px,py))[1], Img.getpixel((px,py))[2]

                    if (R > 10 or G > 10 or B > 10):
                        Visit[py][px] = 1
                        Q.put((px, py))
                        L.append((px, py))

    return L


    
def Complete_CCA2(Img, dim_x, dim_y, Visit): # Make Not Largest Component Set
    
    Complete_Components = []

    for dx in range(dim_x):
        for dy in range(dim_y):
            R, G, B = Img.getpixel((dx,dy))[0], Img.getpixel((dx,dy))[1], Img.getpixel((dx,dy))[2]

            if((R > 10 or G > 10 or B > 10) and (Visit[dy][dx] == 0)): # 영상에서 R,G,B가 모두 0 즉, 까만색 이면서 방문안했던 pixel
                Component_Set = Basic_CCA2(Img, dx, dy, Visit) # dx, dy가 Initial point
                
                if(len(Component_Set) is not 0):
                    Non_Empty_Component = Component_Set
                    Non_Empty_Component = np.array(Non_Empty_Component)
                    Complete_Components.append(Non_Empty_Component)

    return Complete_Components
#%%


#%%
def Save_File(path):

    Img = Image.open(path)
    dim_x, dim_y = np.shape(Img)[0], np.shape(Img)[1]

    File_name = (path.split("/"))[-1]

    # print(Patient_Doppler_Files[PF_len], dim_x, dim_y)
    Clustered = [[[0 for channel in range(3)] for height in range(dim_y)] for width in range(dim_x)]
    Visit = [[0 for width in range (dim_y)] for height in range (dim_x)] # 434, 634
    
    NP_Img = np.array(Img)
    PIL_Img = Image.fromarray(np.uint8(NP_Img))
    print('[',all_data.index(path),'/',len(all_data),']',path)

    CCC = Complete_CCA2(PIL_Img, dim_y, dim_x, Visit)

    Len_of_Max_Components = 0
    for C_len in range(len(CCC)):
        nComponents = CCC[C_len]
        if(Len_of_Max_Components < len(nComponents)):
            Len_of_Max_Components = len(nComponents)

    for C_len in range(len(CCC)):
        nComponents = CCC[C_len]
        if(len(nComponents) == Len_of_Max_Components):
            for CC_len in range(len(nComponents)):
                C_x, C_y = nComponents[CC_len][0], nComponents[CC_len][1]

                Clustered[C_y][C_x][0] = PIL_Img.getpixel((C_x, C_y))[0]
                Clustered[C_y][C_x][1] = PIL_Img.getpixel((C_x, C_y))[1]
                Clustered[C_y][C_x][2] = PIL_Img.getpixel((C_x, C_y))[2]

    Save_Path = path.replace('femoral artery/2022.01.21_KH','femoral_artery_with_KH/Results/CCA_220204_3')

    Clustered = Image.fromarray(np.uint8(Clustered))
    Clustered.save(Save_Path )


 
if __name__ == '__main__':
    with Pool() as p:
        p.map(Save_File, all_data)
        p.close()
        p.join()
       

# %%
all_data[0]
# %%
