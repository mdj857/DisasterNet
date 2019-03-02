# This script combines the human-coder .csvs, cross references them with the candidate Harvey images, and passes
# each image through the VGG-16 model for feature extraction. 


import pandas as pd
from datetime import datetime  
from datetime import timedelta
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
import pickle

images_path = 'coded_images/'
# combine the .csvs into a single dataframe
combined = pd.read_csv('C1.csv')

print combined.head
# function that gets the date given an offset that's been coded
def get_date(coded_date):
	date = datetime(2017,8,16) + timedelta(days=coded_date)
	date = date.strftime('X%m-X%d-%y').replace('X0','X').replace('X','')
	return date

def get_image_name(image_num):
	return 'Image' + str(image_num) + '.jpg'

def get_relative_file_path(date_num, im_num):
	return get_date(date_num) + '/' + get_image_name(im_num)


x_img=[]


for index, row in combined.iterrows():
	i_num = row['Q3ImageNumber']
	date = row['Q1DateofImage']
	
	try: 
		temp_img = image.load_img(images_path+get_relative_file_path(date, i_num), target_size=(224, 224))
		temp_img = image.img_to_array(temp_img)
		x_img.append(temp_img)
	except:
		print "failed " + str(get_relative_file_path(date, i_num))
 
		
#converting train images to array and applying mean subtraction processing
x_img=np.array(x_img) 
train_img=preprocess_input(x_img)


# loading VGG16 model weights
model = VGG16(weights='imagenet', include_top=False)
# Extracting features from the train dataset using the VGG16 pre-trained model

feat_vecs = model.predict(train_img)
with open("feat_vecs.pkl", "wb") as fp:   #Pickling
	pickle.dump(feat_vecs, fp)