# Reference : https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/

from keras import backend
from keras.datasets.mnist import load_data
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.models import Model
from keras.models import Sequential
from matplotlib import pyplot
from numpy import asarray
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import randint
from numpy.random import randn
from scipy.stats import skew
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import csv
import glob
import numpy as np
import os
import os
import pandas as pd
import tensorflow as tf

features = pd.read_csv("D:/HAR/DSADS/features.csv")

# This source code is SGAN for DSADS dataset. 
# To use a different dataset, only modify the feature name of f_col_all list.

LABEL = 'ActivityEncoded'
le = preprocessing.LabelEncoder()
features[LABEL] = le.fit_transform(features['activity'].values.ravel())

features['activity'].value_counts()

f_col_all = ['T_xacc_mean', 'T_xacc_max', 'T_xacc_min','T_xacc_var',
       'T_xacc_std', 'T_xacc_skew', 'T_yacc_mean', 'T_yacc_max', 'T_yacc_min',
       'T_yacc_var', 'T_yacc_std', 'T_yacc_skew', 'T_zacc_mean', 'T_zacc_max',
       'T_zacc_min', 'T_zacc_var', 'T_zacc_std', 'T_zacc_skew', 'T_xgyro_mean',
       'T_xgyro_max', 'T_xgyro_min', 'T_xgyro_var', 'T_xgyro_std',
       'T_xgyro_skew', 'T_ygyro_mean', 'T_ygyro_max', 'T_ygyro_min',
       'T_ygyro_var', 'T_ygyro_std', 'T_ygyro_skew', 'T_zgyro_mean',
       'T_zgyro_max', 'T_zgyro_min', 'T_zgyro_var', 'T_zgyro_std',
       'T_zgyro_skew', 'T_xmag_mean', 'T_xmag_max', 'T_xmag_min', 'T_xmag_var',
       'T_xmag_std', 'T_xmag_skew', 'T_ymag_mean', 'T_ymag_max', 'T_ymag_min',
       'T_ymag_var', 'T_ymag_std', 'T_ymag_skew', 'T_zmag_mean', 'T_zmag_max',
       'T_zmag_min', 'T_zmag_var', 'T_zmag_std', 'T_zmag_skew']
       
X = features1[f_col_all]  # all features

X_OneHotEncoded = pd.get_dummies(X)  # all features and OneHotEncoded
f_col_OHE = list(X_OneHotEncoded.columns.values)

y = features1["ActivityEncoded"].apply(lambda x: 1 if x== "Yes" else 0 )  # Labels
y = features1["ActivityEncoded"]

CLASSES = 19
INPUT_SIZE = 54

# custom activation function
def custom_activation(output):
	logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
	result = logexpsum / (logexpsum + 1.0)
	return result

def define_discriminator(n_classes=CLASSES):
  # image input
  in_image = Input(shape=(INPUT_SIZE,))   
  
  fe = Dense(units=128, activation='sigmoid')(in_image)
  fe = LeakyReLU(alpha=0.2)(fe)
  
  fe = Dense(units=128, activation='sigmoid')(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
 
  fe = Dense(units=128, activation='sigmoid')(fe)
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
  d_model.summary()
  return d_model, c_model

# define the standalone generator model
def define_generator(latent_dim, n_outputs=INPUT_SIZE):
	model = Sequential()
	model.add(Dense(128, activation='sigmoid', kernel_initializer='he_uniform', input_dim=latent_dim))
	model.add(Dense(128, activation='sigmoid'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.summary()
	return model

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


# load the images
def load_real_samples(X,y):    
	X = X / 20
	print(X.shape, y.shape)
	return [X, y]

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

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, c_model, latent_dim, dataset, acc_list, n_samples=100):
    
	# evaluate the classifier model
	X, y = dataset
	_, acc = c_model.evaluate(X, y, verbose=0)
	print('Classifier Accuracy: %.3f%%' % (acc * 100))
	acc_list.append(acc)
    
	# save the generator model
	filename2 = 'models/g_model_%04d.h5' % (step+1)
	g_model.save(filename2)
    
	# save the classifier model
	filename3 = 'models/c_model_%04d.h5' % (step+1)
	c_model.save(filename3)
	#print('>Saved:  %s and %s' % (filename2, filename3))

d_supervised_losses=[]
g_supervised_losses=[]
c_accuray=[]
iteration_checkpoints=[]


# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, dataset, dataset2, latent_dim, acc_list, n_epochs=200, n_batch=100):
    
	# select supervised dataset
	X_sup, y_sup = select_supervised_samples(dataset2)
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
			summarize_performance(i, g_model, c_model, latent_dim, dataset, acc_list)
			d_supervised_losses.append(d_loss1)
			g_supervised_losses.append(g_loss)
			c_accuray.append(c_acc)
			iteration_checkpoints.append(i+1)            

#accuracy list for each epochs
acc_list = []
latent_dim = 100

d_model, c_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)

dataset = load_real_samples(X2, y2)

X, y = dataset
print("Total dataset: ", X.shape, y.shape)

import time
start_time = time.time()
print("Trainig Start")
train(g_model, d_model, c_model, gan_model, dataset, dataset2, latent_dim, acc_list) 
print("End")
print("Time: {:.1f}min".format(((time.time() - start_time))/60))

import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('C:/Users/GC/models/c_model_100000.h5')

_, train_acc = model.evaluate(X_train, y_train, verbose=0)
print('Train Accuracy: %.3f%%' % (train_acc * 100))

_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f%%' % (test_acc * 100))

