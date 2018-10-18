# This script combines the human-coder .csvs, cross references them with the candidate Harvey images, and passes
# each image through the VGG-16 model for feature extraction. 


import pandas as pd
from datetime import datetime  
from datetime import timedelta 

images_path = 'coded_images/'
# combine the .csvs into a single dataframe
coder1 = pd.read_csv('Coder 1 CSV File.csv')
coder2 = pd.read_csv('Coder 2 CSV File.csv')
combined = coder1.append(coder2) 

# function that gets the date given an offset that's been coded
def get_date(coded_date):
	date = datetime(2017,8,16) + timedelta(days=coded_date)
	date = date.strftime('X%m-X%d-%Y').replace('X0','X').replace('X','')
	return date

def get_image_name(image_num):
	return 'Image' + str(image_num) + '.jpg'

def get_relative_file_path(date_num, im_num):
	return get_date(date_num)


#preparing the dataset -- assembling the images and their relevant labels into an x and y
df.apply (lambda row: label_race (row),axis=1)


x_img=[]
for index, row in combined.iterrows():
	image_number = row['Q3ImageNumber']
    temp_img=image.load_img(images_path+,target_size=(224,224))
    temp_img=image.img_to_array(temp_img)
    train_img.append(temp_img)

#converting train images to array and applying mean subtraction processing
x_img=np.array(x_img) 
train_img=preprocess_input(x_img)


# loading VGG16 model weights
model = VGG16(weights='imagenet', include_top=False)
# Extracting features from the train dataset using the VGG16 pre-trained model

feat_vecs = model.predict(train_img)
with open("feat_vecs.pkl", "wb") as fp:   #Pickling
	pickle.dump(feat_vecs, fp)

