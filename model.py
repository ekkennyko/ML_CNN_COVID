import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array

img_size = (224,224)
dir_name = 'COVID-19_Radiography_Dataset/COVID'
img_list = glob.glob(dir_name + '/*')
list_covid = []
for img in img_list:
    temp_img = load_img(img,grayscale=True,target_size=(img_size))
    temp_img_array = img_to_array(temp_img) /255
    list_covid.append(temp_img_array)
list_covid = np.array(list_covid)
list_covid2 = list_covid.reshape(-1,50176)
df_covid=pd.DataFrame(list_covid2)
df_covid['label'] = np.full(df_covid.shape[0],2)
print(df_covid.shape)

img_size = (224,224)
dir_name2 = 'COVID-19_Radiography_Dataset/Normal'
img_list2 = glob.glob(dir_name2 + '/*')
list_normal = []
for img in img_list2:
    temp_img = load_img(img,grayscale=True,target_size=(img_size))
    temp_img_array = img_to_array(temp_img) /255
    list_normal.append(temp_img_array)
list_normal = np.array(list_normal)
list_normal2 = list_normal.reshape(-1,50176)
df_normal=pd.DataFrame(list_normal2)
df_normal['label'] = np.full(df_normal.shape[0],0)
print(df_normal.shape)


img_size = (224,224)
dir_name3 = 'COVID-19_Radiography_Dataset/Viral Pneumonia'
img_list3 = glob.glob(dir_name3 + '/*')
list_others = []
for img in img_list3:
    temp_img = load_img(img,grayscale=True,target_size=(img_size))
    temp_img_array = img_to_array(temp_img) /255
    list_others.append(temp_img_array)
list_others = np.array(list_others)
list_others2 = list_others.reshape(-1,50176)
df_others=pd.DataFrame(list_others2)
df_others['label'] = np.full(df_others.shape[0],1)
print(df_others.shape)