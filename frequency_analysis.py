from __future__ import unicode_literals
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import Counter
import sys

# fix weird unicode bug and convert all strings to unicode
reload(sys)  
sys.setdefaultencoding('utf8')

'''
This function plots the top n most frequent tags for a given date using matplotlib
and saves the figure to the current working directory as <date_freq_analysis.png>.
'''
def plot_frequency_of_top_n_words(df, n, date='ALL'):
	tags_for_date = []
	if (date == 'ALL'):
		for index, row in df.iterrows(): 
			tags_for_date += row['tags'].split(', ') 
	else: 
		# get portion of dataframe whose date is as selected
		date_df =  df[df.date==date]

		# collect all comma-seperated tags for this specific date
		for index, row in date_df.iterrows(): 
			tags_for_date += row['tags'].split(', ') 

	
	# get the n most-common tags in the form: 
	# [(<string1>, <num_occurences>), (<string2>, <num_occurences>)....]
	n_most_common = Counter(tags_for_date).most_common(n)

	tags = zip(*n_most_common)[0]
	count = zip(*n_most_common)[1]
	x_pos = np.arange(len(tags)) 
    
    # plot magic
	plt.bar(x_pos, count,align='center')
	plt.xticks(x_pos, tags, rotation = 'vertical') 
	plt.xlabel('Tags')
	plt.ylabel('Frequency')
	plt.title('Tag Frequency for ' + date)
	plt.gca().margins(x=0)
	plt.gcf().canvas.draw()
	tl = plt.gca().get_xticklabels()
	plt.draw()
	maxsize = max([t.get_window_extent().width for t in tl])
	m = 0.5 # inch margin
	s = maxsize/plt.gcf().dpi*len(tags)+2*m
	margin = m/plt.gcf().get_size_inches()[0]
	plt.gcf().subplots_adjust(left=margin, right=1.-margin, bottom=0.1)
	plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
	plt.tight_layout()
	plt.savefig(date+ str('_freq_analysis')+".png")
	plt.clf()

def main(): 
	df = pd.read_csv("google_cloud_image_annotations.csv")
	# remove the enclosing quotes and trailing comma used to format the .cs
	df['tags'] = df.apply(lambda row: row['tags'].replace(', "', ''), axis=1)
	df['tags'] = df.apply(lambda row: row['tags'].replace('"', ''), axis=1)


	# for date in df.date.unique():
	# 	plot_frequency_of_top_n_words(df, 50, date)

	# plot aggregate analysis 
	plot_frequency_of_top_n_words(df, 100)

if __name__ == "__main__":
    main()


