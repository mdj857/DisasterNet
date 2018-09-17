import io
import glob, os
from google.cloud import vision
from google.cloud.vision import types

'''
Given an Image (google.cloud.vision.types.Image), this function returns all textual 
descriptions of labels 
'''
def get_labels(img):
	# label detection
	response = vision_client.label_detection(image=img)
	return response.label_annotations	

'''
I'll just grab the images from my machine. I'm not going to use the Box API
since they're stored locally on my machine like: 
/Images
	/08-17-2017/
		/image1
		/image2
		...
	...

To make it easy on myself, I'll create a dictionary of the form <image_name: <list_of_labels>>
then just format the dictionary into a .csv

'''
label_dict = {}
vision_client = vision.ImageAnnotatorClient()

for dir, subdir, files in os.walk('/Users/mattjohnson/Desktop/Research/Images'):
	for file in files:
		file_name = os.path.join(dir, file)
		if ('.jpg' not in file_name or os.path.getsize(file_name) == 0):
			continue
		print file_name
		with io.open(file_name, 'rb') as image_file:
			content = image_file.read()
		image = types.Image(content=content)

		# create dict of the form <image_name: <list o' labels>>
		label_dict['file_name'] = get_labels(image)

# write to .csv...
with open('google_cloud_image_annotations.csv', 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in label_dict.items()]
