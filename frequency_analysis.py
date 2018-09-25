from __future__ import unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import sys

'''
This function plots the top n most frequent tags for a given date. 
'''
def plot_frequency_of_top_n_words(date, n):
	# get portion of dataframe whose date is as selected
	date_df =  df[df.date==date]

	# collect all comma-seperated tags for this specific date
	tags_for_date = []
	for index, row in date_df.iterrows(): 
		tags_for_date += row['tags'].split(', ') 

	
	# get the n most-common tags in the form: 
	# [(<string1>, <num_occurences>), (<string2>, <num_occurences>)....]
	n_most_common = Counter(tags_for_date).most_common(n)

	tags = zip(*n_most_common)[0]
	count = zip(*n_most_common)[1]
	x_pos = np.arange(len(tags)) 
    
	plt.bar(x_pos, count,align='center')
	plt.xticks(x_pos, tags, rotation = 'vertical') 
	
	plt.grid()

	plt.gca().margins(x=0)
	plt.gcf().canvas.draw()
	tl = plt.gca().get_xticklabels()
	maxsize = max([t.get_window_extent().width for t in tl])
	m = 0.2 # inch margin
	s = maxsize/plt.gcf().dpi*N+2*m
	margin = m/plt.gcf().get_size_inches()[0]

	plt.gcf().subplots_adjust(left=margin, right=1.-margin)
	plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
	plt.ylabel('Frequency')
	plt.savefig(date+ str('_freq_analysis')+".png")
	plt.show()


	
	

	# # Plot histogram 
	# indexes = np.arange(len(tags))
	# width = 0.7
	# plt.bar(indexes, tag_counts, width, orientation='vertical')
	# plt.xticks(indexes + width * 0.5, tags)
	# plt.show()


# fix weird unicode bug and convert all strings to unicode
reload(sys)  
sys.setdefaultencoding('utf8')


df = pd.read_csv("google_cloud_image_annotations.csv")

# remove the enclosing quotes and trailing comma used to format the .cs
df['tags'] = df.apply(lambda row: row['tags'].replace(', "', ''), axis=1)
df['tags'] = df.apply(lambda row: row['tags'].replace('"', ''), axis=1)

plot_frequency_of_top_n_words('2017-08-17', 50)



