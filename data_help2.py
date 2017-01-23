#encoding="utf-8"
#!/usr/bin/env python

import os
import re
import sys
import json
import jieba
import pickle
import logging
import itertools
import math
import random
import unicodedata
import numpy as np
import pandas as pd
import gensim as gs
from pprint import pprint
from collections import Counter
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			if start_index == end_index:
				continue
			yield shuffled_data[start_index:end_index]

def CalInforGain(postiveNum, negativeNum, totalNum):
	postiveFloat = postiveNum * 1.0 / totalNum
	negativeFloat = negativeNum * 1.0 / totalNum

	if postiveFloat != 0:
		postiveValue = (- 1) * math.log(postiveFloat) * postiveFloat 
	if negativeFloat != 0:
		negativeValue = (- 1) * math.log(negativeFloat) * negativeFloat    # we don't user the calculation of log()/log(2), we use the e instead
	return negativeValue + postiveValue

def CalMutualInfor(postiveNum, negativeNum, totalNum, postiveLen, negativeLen) :
	postMIScore = 10000 * postiveNum * 1.0 / (totalNum * postiveLen)
	negaMIScore = 10000 * negativeNum * 1.0 / (totalNum * negativeLen)
	return postMIScore, negaMIScore

def getPMIandIG(postiveword, negativeword):
	wordInforGain = dict()            # get the information gain of the words
	wordMutualInfor = dict()                   # get the result of the mutual information
	Vocabulay = set(postiveword.keys()) & set(negativeword.keys())

	postiveLen = len(postiveword)   # calculate length of each dict 
	negativeLen = len(negativeword)

	for word in Vocabulay:
		postiveNum = postiveword.get(word, 0)
		negativeNum = negativeword.get(word, 0)
		totalNum = postiveNum + negativeNum 
		if totalNum < 3:
			continue

		wordIG = CalInforGain(postiveNum, negativeNum, totalNum)
		postMIScore, negaMIScore = CalMutualInfor(postiveNum, negativeNum, totalNum, postiveLen, negativeLen)

		wordMutualInfor[word] = postMIScore / negaMIScore #change the value
		wordInforGain[word] = wordIG

	return wordInforGain, wordMutualInfor

def calValue(positive, negative):
	postiveword = dict()
	negativeword = dict()

	for sentence in positive:
		words = {word for word in sentence if not re.match(r'.*(\w)+.*', word)}	
		for word in words:
			postiveword[word] = postiveword.get(word, 0) + 1

	for sentence in negative:
		words = {word for word in sentence if not re.match(r'.*(\w)+.*', word)}	
		for word in words:
			negativeword[word] = negativeword.get(word, 0) + 1

	return postiveword, negativeword

def getValue(x_train, y_train):
	postiveword, negativeword = calValue(x_train, y_train)
	wordInforGain, wordMutualInfor = getPMIandIG(postiveword, negativeword)
	return 	wordInforGain, wordMutualInfor

def lodaModel(w2vModelPath):
	wordModel = dict()
	with open(w2vModelPath, 'r') as f :
		for line in f:
			fields = line.strip().split("\t")
			word = fields[ 0 ]
			vector = map(float, fields[1:])
			if len(vector) == 300 :
				wordModel[word] = vector
	return wordModel

def clean_str(string):
	# string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", "n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()

def load_data2(w2vmodel_file, dev_size, forced_NUM_STEPS=None):
	wordModel = lodaModel(w2vmodel_file)

	positive = list()
	positive_labels = list()
	with open("./data/rt-polaritydata/rt-polarity.pos", "r") as f:
		for line in f:
			line = line.strip()
			line = clean_str(line)
			fields = list(line.split(" "))
			positive.append(fields)
			positive_labels.append([0,1])

	negative = list()
	negative_labels = list()
	with open("./data/rt-polaritydata/rt-polarity.neg", "r") as f:
		for line in f:
			line = line.strip()
			line = clean_str(line)
			fields = list(line.split(" "))
			negative.append(fields)
			negative_labels.append([1,0])

	print "positive is {}, negative is {}, positive_labels is {}, negative_labels is {}".format(len(positive), len(negative), len(positive_labels), len(negative_labels))
	text = positive + negative
	wordset = {word for sentence in text  for word in sentence }
	for word in wordset:
		if word not in wordModel.keys():
			vector = [0.0 for i in range(300) ]
			# vector = [random.uniform(-1000, 1000) * 1.0/1000 for i in range(300) ]
			wordModel[word] = vector

	wordInforGain, wordMutualInfor = getValue(positive, negative)
	for word, vector in wordModel.items():
		if word in wordInforGain.keys():
			value = wordInforGain.get(word)
			vector.append(value)
			wordModel[word] = vector
		else:
			vector.append(0.0)
			wordModel[word] = vector

	for word, vector in wordModel.items():
		if word in wordMutualInfor.keys():
			value = wordMutualInfor.get(word)
			vector.append(value)
			wordModel[word] = vector
		else:
			vector.append(0.0)
			wordModel[word] = vector

	
	#change the chinese to digit that was one to one
	worddict = {w:(i+1) for i,w in enumerate( wordModel.keys() )}
	text = [[worddict.get(word, -1) for word in sentence if worddict.get(word, -1) > 0] for sentence in text]

	if forced_NUM_STEPS is None:
		sequence_length = max([len(sentence) for sentence in text])
	else:
		sequence_length = forced_NUM_STEPS
	print "the sequence length is {}".format(sequence_length)

	#pad the sentence into the same sequence
	x_text = list()
	for sentence in text:
		if len(sentence) > sequence_length:
			x_text.append(sentence[: sequence_length])
		else:
			sentence.extend([0] * (sequence_length - len(sentence)))
			x_text.append(sentence)
	print "The length of x_text is {}".format(len(x_text))

	sort_worddict = sorted(worddict.iteritems(), key=lambda x:x[1], reverse=False)
	w2vModel = list()
	w2vModel.append([0.0] * 302)

	for item in sort_worddict:
		w2vModel.append(wordModel.get(item[0]))
	print "The length of vocalbulary is {}".format(len(w2vModel))

	#change the label to softmax format
	embedding_mat = np.array(w2vModel, dtype = np.float32)

	positiveSample = x_text[:5331]
	negativeSample = x_text[5331:]

	validata_size = - 1 * int(5331 * dev_size)

	#just for evaluation convienent
	train1, dev1 = positiveSample[: validata_size], positiveSample[validata_size: ]
	train2, dev2 = negativeSample[: validata_size], negativeSample[validata_size: ]

	label1_train, labe1_dev = positive_labels[: validata_size], positive_labels[validata_size: ]
	label2_train, label2_dev = negative_labels[: validata_size], negative_labels[validata_size: ]

	train1.extend(train2)
	dev1.extend(dev2)
	label1_train.extend(label2_train)
	labe1_dev.extend(label2_dev)

	print "x_train is {}, y_train is {}, x_dev is {}, y_dev is {}".format(len(train1), len(label1_train), len(dev1), len(labe1_dev))

	return np.array(train1), np.array(label1_train), np.array(dev1), np.array(labe1_dev), embedding_mat

if __name__ == "__main__":
	w2vmodel_file = "./data/googleVector.bin"
	load_data2(w2vmodel_file, 0.1, forced_NUM_STEPS=50)