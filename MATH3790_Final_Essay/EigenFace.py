import cv2
import os
from PIL import Image
import numpy as np
import math 
import pandas as pd
import matplotlib.pyplot as plt
import os
def EigenFace(Training_Data):
    meanface = np.mean(Training_Data, axis=1).reshape(10000,1)
    # Find de mean face
    de_meanface=Training_Data-meanface
    # find corvariance
    C=np.dot(de_meanface.T,de_meanface)
    # Find Eigenvalue and Eigenvector
    D,V=np.linalg.eig(C)
    # Find Eigenfaces
    Eigenfaces = np.dot(de_meanface, V)
    return meanface,de_meanface,Eigenfaces

