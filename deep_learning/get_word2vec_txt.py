"""
Script that Converts word2vec bin file to text

Compile command : 
python get_word2vec_txt.py --word2vec ../GoogleNews-vectors-negative300.bin
"""

import argparse
from gensim.models.keyedvectors import KeyedVectors


def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-w", "--word2vec", required=True, \
		help='path to word2vec pre-trained model')
	args = parser.parse_args()

	path_word2vec = args.word2vec

	model = KeyedVectors.load_word2vec_format(path_word2vec, binary=True)
	print("Converting to txt format ...")
	model.save_word2vec_format('../GoogleNews-vectors-negative300.txt', binary=False)
	print("... Done")

if __name__ == '__main__':
	main()