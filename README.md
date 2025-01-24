# week1
Develop an AI-powered waste ﻿classification model using image processing and machine learning techniques . Accurately categorize waste materials based on visual images using the datasset
https://www.kaggle.com/datasets/techsash/waste-classification-data/data

# source code
pip install opencv-python
pip install tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
train_path ="dataset/TRAIN"
test_path= "dataset/TEST"
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore
from glob import glob
#visulization
from cv2 import cvtColor
x_data=[]
y_data=[]
for category in glob(train_path+'/*'):
    for file in tqdm(glob(category+"/*")):
        img_array = cv2.imread(file)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        x_data.append(img_array)
        y_data.append(category.split('/')[-1])
data=pd.DataFrame({'image':x_data,'label':y_data})
data.shape
colors=['#a0d157','#c48bb8']
plt.pie(data.label.value_counts(), labels=['Organic', 'Recyclabel'] ,autopct='%0.2f%%', colors=colors, startangle=90, explode=[0.005,0.005])
plt.show
