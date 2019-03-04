# DisasterNet: Evaluating the Performance of Transfer Learning to Classify Hurricane-Related Images Posted on Twitter

This project focuses on using transfer learning to classify images posted on Twitter during Hurricane Harvey
into various categories useful to first responders such as urgency, relevance, presence of damage in the image. This project is part of an ongoing research effort by the Computational Media Lab at the University of Texas at Austin, supervised by Dr. Dhiraj Murthy.

**NOTE:** In order to access the image database and pre-trained models used in this study, please download the .zip file located [here](https://www.dropbox.com/s/ifqrz7yc2n5byl9/models_and_data.zip?dl=0).


The repository contains three main subdirectories, outlined below: 

## google_cloud_vision
This directory contains scripts used to access Google Cloud Vision's API. To use these scripts, you will need to generate your own API key.
Tags for each image in the codebook were collected and treated as documents in the corpus of the collection of tags for each image. 

The subdirectory /plots contains frequency plots of tags for each type of image. These were primarily created as an exploratory tool to help
understand the content of the image database. 

## graphs_plots_and_figure
This directory contains all plots and figures used in the authoring the sequence of publications submitted under this project. 

## model_scripts_and_evaluation
This directory contains three subdirectories.

**preprocess**
Contains scripts used to preprocess each image and create feature vectors from each image using the VGG-16 convolutional neural network.

**model_training**
Contains scripts used to create and train each model from a pickled feature vector file and the codebook .csv file.

**f1_score_scripts**
Contains scripts used to compute the F1-Micro and F1-Macro score of each classifier from their corresponding .h5 file. 

If you have any questions, please don't hesitate to send an email to
mjohn son082396 -at- gmail.com
