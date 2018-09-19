import io
import glob, os
import csv
from google.cloud import vision
from google.cloud.vision import types

'''
Given an Image (google.cloud.vision.types.Image), this function returns all textual 
descriptions of labels 
'''
def get_labels(img):
	# label detection
	response = vision_client.label_detection(image=img)
	labels = response.label_annotations	
	descriptions = '"'
	for label in labels:
		descriptions+= label.description + ', '
 	descriptions += '"'
	descriptions = descriptions.encode('utf-8').strip()
	return descriptions

'''
I'll just grab the images from my machine. I'm not going to use the Box API
since they're stored locally on my machine like: 
/Images
	/08-17-2017/
		/image1
		/image2
		...
	...

'''
images_list = []
vision_client = vision.ImageAnnotatorClient()

# get all file paths in the directory
for dir, subdir, files in os.walk('/Users/mattjohnson/Desktop/Research/Images'):
	for file in files:
		file_name = os.path.join(dir, file)
		if ('.jpg' not in file_name or os.path.getsize(file_name) == 0 or '.csv' in file_name):
			continue
		print file_name
		with io.open(file_name, 'rb') as image_file:
			content = image_file.read()
		try:
			image = types.Image(content=content)
		except: 	
			print "ERROR WITH " + file_name
			# write to .csv...
			with open('google_cloud_image_annotations.csv', 'w') as f:
				mywriter = csv.writer(f)
				mywriter.writerow(['image', 'tags'])
				for d in images_list:
					mywriter.writerow(d)


		# create list of of the form [image_name, string of labels] to resemble one image>
		images_list.append([file_name,get_labels(image)])
		

# write to .csv...
with open('google_cloud_image_annotations.csv', 'w') as f:
	mywriter = csv.writer(f)
	mywriter.writerow(['image', 'tags'])
	for d in images_list:
		mywriter.writerow(d)