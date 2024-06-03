import cv2
import os
from PIL import Image
import numpy as np
import math 
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import FaceRead as RF
import EigenFace as EF
import FaceRecognition as FR
height =100
width=100
train_fig_num=51
# Training_Path = 'E:/2024 Spring/MATH 3790/Assignment/Final project/TrainDatabase' 
Training_Path='E:/2024 Spring/MATH 3790/Assignment/Final project 2/Training_1'
# Training_Path='E:/2024 Spring/MATH 3790/Assignment/Final project 2/User_Face'
files = os.listdir(Training_Path)
Training_Data = RF.face_space(Training_Path,width,height,train_fig_num)
print(Training_Data.shape)
# TestImage='C:/Users/36955/Desktop/Zoran_Djindjic_0001.jpg'
# TestImage='E:/2024 Spring/MATH 3790/Assignment/Final project/TrainDatabase/Aaron_Guiel_0001.jpg'
TestImage='E:/2024 Spring/MATH 3790/Assignment/Final project 2/User_Face/Gale W.png'
ti=RF.open_convert_image(TestImage,width,height)
m,A,Eigenfaces = EF.EigenFace(Training_Data)
#Save eigenface data
# file_path1 = "E:/2024 Spring/MATH 3790/Assignment/Final project 2/EigenFaceData.xlsx"
# pd.DataFrame(Eigenfaces).to_excel(file_path1, index=False, header=False)
# print("Array has been written to", file_path1)
# print(A.shape,Eigenfaces.shape)

OutputName=FR.FaceRecognition(ti,m,A,Eigenfaces)
print(OutputName)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

# use for loop to fill each subfigures
# Print part of eigen face
for i in range(4):
    for j in range(4):
        p = i * 4 + j  # According to the position of subfigures to calculatethe corresponde index
        ax = axes[i, j]
        ax.imshow(Eigenfaces[:, p].reshape(100, 100), cmap='gray')
        ax.set_xticks([])  #hide x axis 
        ax.set_yticks([])  #hide y axis

# adjust subfigures distance and layout
plt.tight_layout()
plt.show()