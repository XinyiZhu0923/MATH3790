import cv2
import os
from PIL import Image
import numpy as np
import math 
import pandas as pd
import matplotlib.pyplot as plt

#Open the image in grey base
def open_convert_image(url,width,height):
    image=cv2.imread(url)
    resized_image = cv2.resize(image, (width,height), interpolation=cv2.INTER_AREA)
    gray_image=cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return gray_image.flatten()

#Finde the face space x
def face_space(path,width,height,train_fig_num):
    disserted_matric=np.zeros((width*height,train_fig_num))
    i=0
    files = os.listdir(path)
    for file in files:
        file_path=os.path.join(path,file)
        img=open_convert_image(file_path,width,height)
        disserted_matric[:,i]=img
        i=i+1
    return disserted_matric