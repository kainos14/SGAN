#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


# In[2]:


features = pd.read_csv("D:/HAR/PAMAP2_Dataset/PAMAP_features.csv",index_col = 0)


# In[3]:


features


# In[4]:


features['activity_id'].value_counts()


# In[5]:


from IPython.display import Image 

Image("act1.png")


# In[6]:


activity_id = {0: 'transient', 1:'lying', 2:'sitting', 3:'standing',
              4:'walking', 5:'running', 6:'cycling', 7:'Nordic walking',
              9:'watching TV', 10:'computer work', 11:'car driving',
              12:'ascending stairs', 13:'descending stairs', 16:'vacuum cleaning',
              17:'ironing', 18:'folding laundry', 19:'house cleaning',
              20:'playing soccer', 24:'rope jumping'}

protocol_acts = [1,2,3,4,5,6,7,17,16,24,12,13]


# In[7]:


s = features.groupby('activity_id').count()['act_level']
s = s.rename("Activity Counts")
s.index = [activity_id[x] for x in protocol_acts]
print(('Dev Dataset by Activity'))
display(s.sort_values(ascending =False))
ax = s.sort_values(ascending =False).plot(kind='bar', figsize=(8,4))
_ = ax.set_ylabel('windows')
_ = ax.set_xlabel('activity')
_ = ax.set_title('Test Dataset by Activity') 


# In[8]:


features['activity_id'].value_counts()


# In[9]:


features['sub_id'].value_counts()


# In[10]:


features['activity_id'].value_counts()


# In[11]:


'''
idx = features[features['activity_id'] == 17].index
features.drop(idx , inplace=True)
idx = features[features['activity_id'] == 4].index
features.drop(idx , inplace=True)
idx = features[features['activity_id'] == 7].index
features.drop(idx , inplace=True)
idx = features[features['activity_id'] == 16].index
features.drop(idx , inplace=True)
idx = features[features['activity_id'] == 6].index
features.drop(idx , inplace=True)
idx = features[features['activity_id'] == 5].index
features.drop(idx , inplace=True)
idx = features[features['activity_id'] == 12].index
features.drop(idx , inplace=True)
idx = features[features['activity_id'] == 13].index
features.drop(idx , inplace=True)
idx = features[features['activity_id'] == 24].index
features.drop(idx , inplace=True)
'''


# In[12]:


label=LabelEncoder()
features['activity_id']=label.fit_transform(features['activity_id'])
features.head()


# In[13]:


features['activity_id'].value_counts()


# In[14]:


features['sub_id'].value_counts().plot(kind = "bar",figsize = (12,6))
plt.show()


# In[15]:


features.columns


# In[16]:


'''
condition = np.where(features['sub_id'] == 6)
sample_features = features.iloc[condition]

condition2 = np.where(features['sub_id'] != 6)
features = features.iloc[condition2]
'''


# In[17]:


features['activity_id'].value_counts()


# In[18]:


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
        
X_train = features[f_col_all]  # all features

X_OneHotEncoded = pd.get_dummies(X_train)  # all features and OneHotEncoded
f_col_OHE = list(X_OneHotEncoded.columns.values)

y_train = features["activity_id"].apply(lambda x: 1 if x== "Yes" else 0 )  # Labels
y_train = features["activity_id"]


# In[19]:


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
        
X_test = features[f_col_all]  # all features

X_OneHotEncoded = pd.get_dummies(X_test)  # all features and OneHotEncoded
f_col_OHE = list(X_OneHotEncoded.columns.values)

y_test = features["activity_id"].apply(lambda x: 1 if x== "Yes" else 0 )  # Labels
y_test = features["activity_id"]


# In[20]:


#X_train, X_test, y_train, y_test = train_test_split(X_OneHotEncoded, y, test_size=0.25, random_state=6)

print("X_train :", X_train.shape)
print("y_train :", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# In[21]:


from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv1D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from matplotlib import pyplot
from keras import backend
from keras.models import Sequential


# In[22]:


CLASSES = 12
INPUT_SIZE = X_train.shape[1]
INPUT_SIZE 


# In[23]:


# custom activation function
def custom_activation(output):
	logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
	result = logexpsum / (logexpsum + 1.0)
	return result


# In[24]:


NEURON = 256


# In[25]:


def define_discriminator(n_classes=CLASSES):
  # image input
  in_image = Input(shape=(INPUT_SIZE,))   
  # downsample
  fe = Dense(units=NEURON , activation='sigmoid')(in_image)
  fe = LeakyReLU(alpha=0.2)(fe)
  # downsample
  fe = Dense(units=NEURON , activation='sigmoid')(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  # downsample
  fe = Dense(units=NEURON , activation='sigmoid')(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  # dropout
  fe = Dropout(0.4)(fe)
  # output layer nodes
  fe = Dense(n_classes)(fe)
  # supervised output
  c_out_layer = Activation('softmax')(fe)

  # define and compile supervised discriminator model
  
  c_model = Model(in_image, c_out_layer)
  c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.003, beta_1=0.5), metrics=['accuracy'])

  # unsupervised output
  d_out_layer = Lambda(custom_activation)(fe)

  # define and compile unsupervised discriminator model
  d_model = Model(in_image, d_out_layer)
  d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.003, beta_1=0.5))    
  print("-------------\n")    
  print("discriminator\n")
  print("-------------\n")    
  d_model.summary()
  return d_model, c_model


# In[26]:


# define the standalone generator model
def define_generator(latent_dim, n_outputs=INPUT_SIZE):
	model = Sequential()
	model.add(Dense(200, activation='sigmoid', kernel_initializer='he_uniform', input_dim=latent_dim))
	model.add(Dense(NEURON , activation='sigmoid'))
	model.add(Dense(n_outputs, activation='relu'))
	print("-------------\n")    
	print("generator\n")
	print("-------------\n")       
	model.summary()
	return model


# In[27]:



# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect image output from generator as input to discriminator
	gan_output = d_model(g_model.output)
	# define gan model as taking noise and outputting a classification
	model = Model(g_model.input, gan_output)
	# compile model
	opt = Adam(lr=0.003, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)

	return model


# In[28]:


# load the images
def load_real_samples(X,y):    
	X = X / 20
	print(X.shape, y.shape)
	return [X, y]


# In[29]:


# select a supervised subset of the dataset, ensures classes are balanced

def select_supervised_samples(dataset2, n_classes=CLASSES):
  X, y = dataset2
  n_samples = X.shape[0] *0.03
  print("Sample :", n_samples)
  rand = randint(0,1000)
  #n_per_class = int(n_samples / n_classes)
  n_per_class = int(n_samples)
  return X.sample(n=n_per_class,replace=False, random_state=rand), y.sample(n=n_per_class,replace=False, random_state=rand)


# In[30]:


# select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
  features, labels = dataset
	# choose random instances
  rand = randint(0, 1000)
	# select features and labels
  X = features.sample(n=n_samples,random_state=rand)
  labels = labels.sample(n=n_samples,random_state=rand)
	# generate class labels
  y = ones((n_samples, 1))
  return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	z_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = z_input.reshape(n_samples, latent_dim)
	return z_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict(z_input)
	# create class labels
	y = zeros((n_samples, 1))
	return images, y


# In[31]:


# generate samples and save as a plot and save the model

def summarize_performance(step, g_model, c_model, latent_dim, dataset, acc_list, n_samples=100):
    
	# evaluate the classifier model
	X, y = dataset
	_, acc = c_model.evaluate(X, y, verbose=0)
	print('Classifier Accuracy: %.3f%%' % (acc * 100))
	acc_list.append(acc)

    
	# save the generator model
	filename2 = 'C:/Users/GC/models/g_model_%04d.h5' % (step+1)
	#g_model.save(filename2)
    
	# save the classifier model
	filename3 = 'C:/Users/GC/models/c_model_%04d.h5' % (step+1)
	#c_model.save(filename3)
	print('>Saved:  %s and %s' % (filename2, filename3))


# In[32]:


d_supervised_losses=[]
g_supervised_losses=[]
c_accuray=[]
iteration_checkpoints=[]


# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, dataset, dataset2, latent_dim, acc_list, n_epochs=160, n_batch=100):
    
	# select supervised dataset
	X_sup, y_sup = select_supervised_samples(dataset2)
	print("Class Balance: \n", y_sup.value_counts())    
	print("Select supervised dataset: ", X_sup.shape, y_sup.shape)
    
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
    
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
    
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    
	# manually enumerate epochs
	for i in range(n_steps):
		# update supervised discriminator (c)
		[Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
		c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
		# update unsupervised discriminator (d)
		[X_real, _], y_real = generate_real_samples(dataset, half_batch)
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)
		# update generator (g)
		X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		# summarize loss on this batch
		print('>%d, c[%.3f,%.0f%%], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))                

        # evaluate the model performance every so often                        
        
		if (i+1) % (100) == 0:            
			_, train_acc = c_model.evaluate(X, y, verbose=0)            
			print('>>> Accuracy: %.3f%%' % (train_acc * 100))

			summarize_performance(i, g_model, c_model, latent_dim, dataset, acc_list)
			d_supervised_losses.append(d_loss1)
			g_supervised_losses.append(g_loss)
			c_accuray.append(c_acc)
			iteration_checkpoints.append(i+1)            


# In[33]:


#accuracy list for each epochs
acc_list = []

# size of the latent space
latent_dim = 100
# create the discriminator models
d_model, c_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples(X_test, y_test)
X, y = dataset

dataset2 = load_real_samples(X_train, y_train)
X2, y2 = dataset2

print("Total dataset: ", X.shape, y.shape)
print("Sample dataset: ", X2.shape, y2.shape)


# In[34]:


import time
start_time = time.time()
print("Trainig Start")

train(g_model, d_model, c_model, gan_model, dataset, dataset2, latent_dim, acc_list)

print("End")
print("Time: {:.1f}min".format(((time.time() - start_time))/60))


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('C:/Users/GC/models/c_model_4100.h5')

_, train_acc = model.evaluate(X_train, y_train, verbose=0)
print('Train Accuracy: %.3f%%' % (train_acc * 100))

_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f%%' % (test_acc * 100))


# In[ ]:




