#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os


# In[60]:


data= pd.read_csv("D:/HAR/PAMAP2_Dataset/PAMAP_features.csv",index_col = 0)


# In[61]:


data


# In[62]:


data['activity_id'].value_counts()


# In[64]:


label=LabelEncoder()
data['activity_id']=label.fit_transform(data['activity_id'])
data.head()


# In[65]:


NUM_LABEL = 45
SAMPLE_NUM = 10


# In[66]:


def create_samples(data, num_duplicates=NUM_LABEL):
    df = pd.DataFrame()
    for label in data['activity_id'].unique():
        label_data = data[data['activity_id'] == label]
        samples = label_data.sample(n=num_duplicates, replace=True)
        df = pd.concat([df, samples], ignore_index=True)  # 'concat' 사용
    return df


# In[67]:


df['activity_id'].value_counts()


# In[68]:


for i in range(SAMPLE_NUM):
    df = create_samples(data, num_duplicates=NUM_LABEL)

    # Save data as CSV file
    output_csv_file = f'D:/HAR/PAMAP2_Dataset/samples_{i + 1}.csv'
    df.to_csv(output_csv_file, index=False)
    print(f'Repeat {i+1} - data has been saved to file {output_csv_file}.')


# In[76]:


# load model
model = load_model('C:/Users/GC/models/c_model_0400.h5')

# Folder path
data_folder = 'D:/HAR/PAMAP2_Dataset'

# Read and evaluate files 
for i in range(1, SAMPLE_NUM):
     filename = f'D:/HAR/PAMAP2_Dataset/samples_{i}.csv'
     file_path = os.path.join(data_folder, filename)

     # Read data from CSV file
     df = pd.read_csv(file_path)
     print(df.shape)  
     f_col_all = ['hr_mean', 'hr_mean_normal', 'hr_std', 'hr_std_normal', 
                  'hand_acc_x_mean', 'hand_acc_x_std', 'hand_acc_y_mean','hand_acc_y_std', 
                  'hand_acc_z_mean', 'hand_acc_z_std', 'hand_gyr_x_mean', 'hand_gyr_x_std', 
                  'hand_gyr_y_mean', 'hand_gyr_y_std', 'hand_gyr_z_mean', 'hand_gyr_z_std',
                  'chest_acc_x_mean', 'chest_acc_x_std', 'chest_acc_y_mean', 'chest_acc_y_std', 
                  'chest_acc_z_mean', 'chest_acc_z_std', 'chest_gyr_x_mean', 'chest_gyr_x_std',
                  'chest_gyr_y_mean', 'chest_gyr_y_std', 'chest_gyr_z_mean', 'chest_gyr_z_std',
                  'ankle_acc_x_mean', 'ankle_acc_x_std', 'ankle_acc_y_mean', 'ankle_acc_y_std', 
                  'ankle_acc_z_mean', 'ankle_acc_z_std', 'ankle_gyr_x_mean', 'ankle_gyr_x_std',
                  'ankle_gyr_y_mean', 'ankle_gyr_y_std', 'ankle_gyr_z_mean', 'ankle_gyr_z_std']
        
     X_test = df[f_col_all]  # all features

     X_OneHotEncoded = pd.get_dummies(X_test)  # all features and OneHotEncoded
     f_col_OHE = list(X_OneHotEncoded.columns.values)

     y_test = df["activity_id"].apply(lambda x: 1 if x== "Yes" else 0 )  # Labels
     y_test = df["activity_id"]        
     print(X_test.shape)
     print(y_test.shape)  
    
     _, test_acc = model.evaluate(X_test, y_test, verbose=0)
     print('Test Accuracy: %.3f%%' % (test_acc * 100))    

