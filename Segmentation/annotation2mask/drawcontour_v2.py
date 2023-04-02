import os
import json
import numpy as np
from PIL import Image
import random
import cv2
import natsort
import shutil

def draw_circle(img, x, y, Color):
    Radius = 5

    for dx in range(x - Radius, x + Radius):
        for dy in range(y - Radius, y + Radius):
            img[dy][dx][0] = Color[0]
            img[dy][dx][1] = Color[1]
            img[dy][dx][2] = Color[2]

    return img

Data_Path = "//10.10.10.10/nas01/Jayoung/femoral_artery/data2/All_Category_56_check_80/"
Class = os.listdir(Data_Path)
Class = natsort.natsorted(Class)

Data_List = []
Json_List = []

for C_len in range(len(Class)):
    All_Files = os.listdir(Data_Path + Class[C_len] + "/" )
    for F_len in range(len(All_Files)):
        if(".png" in All_Files[F_len] and "LONG" in All_Files[F_len]):
            if os.path.exists(Data_Path + Class[C_len] + "/" + All_Files[F_len].replace('.png', '.json')):
                Data_List.append(Data_Path + Class[C_len] + "/" + All_Files[F_len])
                Json_List.append(Data_Path + Class[C_len] + "/" + All_Files[F_len].replace('.png', '.json'))
print(len(Data_List), len(Json_List))


Save_Path = "//10.10.10.10/nas01/Jayoung/femoral_artery/data2/mask/All_Category_56_check_80/"
for F_len in range(len(Data_List)):
    try:
    # for F_len in range(2):
        print(str(F_len + 1) + "/" + str(len(Data_List)))
        Color_Name = []
        Color_List = []
        VS_Img = Image.open(Data_List[F_len])
        VS_Img_npy = np.array(VS_Img)
        VS_Img_npy = np.stack((VS_Img_npy,)*3, axis=-1)

        Total_GT = np.array([[0 for width in range(np.shape(VS_Img_npy)[1])] for width in range(np.shape(VS_Img_npy)[0])])
        Total_GT_Name = (Data_List[F_len].split('/'))[-1].split('.')[0] + "_TOTAL.png"
        with open(Json_List[F_len], "r") as cxrReport:
            # Annot_Json = json.load(cxrReport)
            Annot_Json = json.load(cxrReport)

        Label_Num = len(Annot_Json['shapes'])
        for L_len in range(Label_Num):
            R, G, B = random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)

            Label_Name = Annot_Json['shapes'][L_len]['label']
            if(Label_Name not in Color_Name):
                Color_Name.append(Label_Name)
                Color_List.append([R, G, B])
                globals()['GT_%s' % Label_Name] = np.array([[0 for width in range(np.shape(VS_Img_npy)[1])] for width in range(np.shape(VS_Img_npy)[0])])
                globals()['GT_Point_%s' % Label_Name] = []
                globals()['GT_Check_%s' % Label_Name] = 0
                globals()['Save_Name_%s' % Label_Name] = (Data_List[F_len].split('/'))[-1].split('.')[0] + "_" + Label_Name + ".png"
                
            Label_Idx = Color_Name.index(Label_Name)
            Color = Color_List[Label_Idx]

            # print(Annot_Json['shapes'][L_len]['label'], Color)

            Points_Set = Annot_Json['shapes'][L_len]['points']
            for P_len in range(len(Points_Set)):
                X, Y = Points_Set[P_len][0], Points_Set[P_len][1]
                X, Y = int(round(X)), int(round(Y))
                Colored_Img = draw_circle(VS_Img_npy, X, Y, Color)

            if(globals()['GT_Check_%s' % Label_Name] == 0):
                for P_len in range(len(Points_Set)):
                    X, Y = Points_Set[P_len][0], Points_Set[P_len][1]
                    X, Y = int(round(X)), int(round(Y))
                    globals()['GT_Point_%s' % Label_Name].append([[X, Y]])
                    globals()['GT_Check_%s' % Label_Name] = 1

            elif(globals()['GT_Check_%s' % Label_Name] == 1):
                for P_len in range(len(Points_Set)):
                    X, Y = Points_Set[len(Points_Set) - P_len - 1][0], Points_Set[len(Points_Set) - P_len - 1][1]
                    X, Y = int(round(X)), int(round(Y))
                    globals()['GT_Point_%s' % Label_Name].append([[X, Y]])
            
                globals()['GT_Point_%s' % Label_Name] = [np.array(globals()['GT_Point_%s' % Label_Name])]
                cv2.drawContours(globals()['GT_%s' % Label_Name], globals()['GT_Point_%s' % Label_Name], 0, 255, -1)
                globals()['GT_Check_%s' % Label_Name] = 2

                # globals()['GT_%s' % Label_Name] = Image.fromarray(np.uint8(globals()['GT_%s' % Label_Name]))
                # globals()['GT_%s' % Label_Name].save(Save_Path + globals()['Save_Name_%s' % Label_Name])

                C_X, C_Y = np.where(np.array(globals()['GT_%s' % Label_Name])[:,:] != 0)
                Total_GT[C_X, C_Y] = 255

        # shutil.copy(Json_List[F_len], Save_Path + (Json_List[F_len].split('/'))[-1])

        Total_GT = Image.fromarray(np.uint8(Total_GT))
        Total_GT.save(Save_Path + Total_GT_Name)
        
        Colored_Img = Image.fromarray(np.uint8(Colored_Img))
        # Colored_Img.save(Save_Path + (Data_List[F_len].split('/'))[-1])
    except:
        print(Total_GT_Name)