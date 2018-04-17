"""
Cleaning the Yelp dataset 
Output : train_df, test_df (pandas DataFrame)

# Compile command :  
python data_cleaning.py --dataset ../yelp_academic_dataset_review.pickle
"""
#python data_cleaning.py --dataset /mnt/agp/ghurtado/sentiment/data/yelp_academic_dataset_review.pickle

# remove warnings
import warnings
warnings.filterwarnings('ignore')

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# making output consistent
seed = 46

# Construct the argument parse and parse the arguments
"""
 --dataset: The path to the Yelp directory residing on disk.
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, \
	help="path to Yelp dataset")
args = parser.parse_args()


# Import data
path_data = args.dataset
data = pd.read_pickle(path_data)

# Removing all ('\n') characters using list comprehensions
data['text'] = [txt.replace('\n', '') for txt in data['text']]

# Taking only text and stars columns
data = data.loc[:, ['text', 'stars']]


# Split train/test data
X, y = data.loc[:, 'text'], data.loc[:, 'stars']

test_size = 0.20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

print("X_train.shape: ", X_train.shape, "y_train.shape: ", y_train.shape)
print("X_test.shape: ", X_test.shape, "y_test.shape: ", y_test.shape)

# Saving to csv
train_df = np.vstack((X_train, y_train))
test_df = np.vstack((X_test, y_test))

train_df= pd.DataFrame({'text': train_df[0, :], 'stars': train_df[1, :]})
test_df = pd.DataFrame({'text': test_df[0, :], 'stars': test_df[1, :]})

train_df.to_csv("train_df.csv", index=False, encoding='utf-8')
test_df.to_csv("test_df.csv", index=False, encoding='utf-8')

print("Done")