#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


# In[2]:


## read data:
import pandas as pd
folder_names = sorted(os.listdir("../Data/ILSVRC/Data/CLS-LOC/train/"))

folder_names =sorted([i for i in folder_names if "n" in i])
print(len(folder_names))

# match class based on alphabet:
label_to_index = dict((name, index) for index, name in enumerate(folder_names))


# In[150]:


### We can select the number of samples we train
# There are 1 million images and we can take 5% percent, which is 50k

N_train = 20000
train_csv = pd.read_csv("../Data/LOC_train_solution.csv")


# In[7]:


## name and synset match
file_synset = open("../Data/LOC_synset_mapping.txt")
synset_match =  pd.DataFrame(columns=["Id","names"])
for f in file_synset:
    temp=f.replace("\n","").replace(",","").split(" ")
    # print(temp)
    s2 = pd.DataFrame(np.atleast_2d([temp[0],temp.pop()]),columns=["Id","names"]) 
    synset_match = pd.concat([synset_match, s2], ignore_index=True)
    


# In[8]:


# read data and box:

data_path = "../Data/ILSVRC/Data/CLS-LOC/train/"
data = pd.DataFrame()
image_path = []
class_all = []
boxes = []



for i in range(N_train):
    
    temp = data_path+train_csv["ImageId"][i].split("_")[0]+"/"
    image_path.append(temp+train_csv["ImageId"][i]+".JPEG")
    class_all.append(train_csv["ImageId"][i].split("_")[0])
    # box from training csv
    temp = [i for i in train_csv["PredictionString"][i].split(" ") if i.isdigit()]

    boxes.append(temp)
    
class_name_array =class_all
class_id = np.array([label_to_index[i] for i in class_all])


image_path = np.array(image_path)


# In[9]:


# save txt
f_script= open("dataset_20k.txt","w+")
for i in range(N_train):
    line = image_path[i]
    line+=" "
    #print(boxes[i])
    temp_i = boxes[i]

    for j in range(int(len(boxes[i])/4)):

        line=line+"{},{},{},{},{}".format(temp_i[4*j],temp_i[4*j+1],temp_i[4*j+2],temp_i[4*j+3],class_id[i])
        line+= " "

    line+="\n"
    #print(line)
    f_script.write(line)



# In[10]:





# In[84]:


class_all


# In[25]:


f_script= open("classes.txt","w+")
for i in range(len(folder_names)):
    line=folder_names[i]
    line+="\n"
    f_script.write(line)


# In[26]:


print(len(class_name_array))


# In[18]:


print(len(folder_names))


# In[29]:


with open("classes.txt") as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]


# In[31]:


print(len(class_names))


# In[28]:


print(len(folder_names))


# In[6]:





# In[ ]:




