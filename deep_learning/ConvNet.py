"""
Script that train a ConvNet model for the Yelp dataset
This is inspired from Yoon Kim paper : 
https://arxiv.org/abs/1408.5882

To execute the script, you need to download the 
Google's pre-trained Word2Vec model available at : 
https://code.google.com/archive/p/word2vec/ 

# Compile command : 
	python ConvNet.py --dataset ../yelp_academic_dataset_review.pickle 
	--word2vec ../GoogleNews-vectors-negative300.bin

# Note : 
You might need to install gensim package : 
pip install gensim 
conda install -c anaconda gensim (anaconda environment)
"""

## Import libraries
# remove warnings
import warnings
warnings.filterwarnings('ignore')

import argparse
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout, Merge
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from matplotlib import pyplot as plt

# Making output consistent
seed = 46

## Construct the argument parse and parse the arguments
"""
	--dataset: The path to the Yelp directory residing on disk
	--word2vec : The path to the word2vec pre-trained model
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, \
	help='path to Yelp dataset')
parser.add_argument("-w", "--word2vec", require=True, \
	help='path to word2vec pre-trained model')
args = parser.parse_args

path_data = args.dataset
path_word2vec = args.word2vec