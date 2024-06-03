import cv2
import os
from PIL import Image
import numpy as np
import math 
import pandas as pd
import matplotlib.pyplot as plt
import os

def FaceRecognition(ti,m,A,Eigenfaces):
    Train_Number = A.shape[1]
    ProjectedImages=np.empty(shape=(Train_Number,Train_Number))
    # Find the projectiion of training set (50,50)
    for i in range(Train_Number):
        temp=np.dot(np.transpose(Eigenfaces),A[:,i].reshape(10000,1)).reshape(Train_Number,)
        ProjectedImages[:,i]=temp
    
    Difference = ti.reshape(10000,1)-m
    # Finde the projection of test image 50,1
    Projected_TestImage = np.dot(np.transpose(Eigenfaces),Difference)
    # Find the diff between two projections
    Euc_dist=ProjectedImages-Projected_TestImage
    for i in range (Train_Number):
        Euc_dist[:,i]=np.linalg.norm(Euc_dist[:,i])**2
    print(np.min(Euc_dist))
    Euc_dist_min = np.min(Euc_dist)
    Recognized_index = np.argmin(Euc_dist)
    if Euc_dist_min<10:
        OutputName = str(Recognized_index) + '.jpg'
    else:
        OutputName="not in the database"
    return OutputName
    # return Projected_TestImage