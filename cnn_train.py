#encoding="utf-8"
#!/usr/bin/env python

import sys
import tensorflow as tf
import numpy as np
import random
import time
import os
import data_help2
import datetime

from text_CNN import TextCNN

reload( sys )
sys.setdefaultencoding( "utf-8" )

"""
# Parameters
# ===============================================
"""

# data load parameter
tf.flags.DEFINE_string( "dataPath" , "./data/travelogueData" , "the data Path" )
tf.flags.DEFINE_string( "stopwordsPath" , "./data/stopwords.txt" , "The stopwwords path" )
tf.flags.DEFINE_string( "w2vModelPath" , "./data/w2vmodel.bin" , "the model of word2vec" )
tf.flags.DEFINE_float( "dev_sample_percentage" , 0.2 , "percentage of the train data to use for test" )
tf.flags.DEFINE_integer( "sequence_length" , 35 , "the max of words in a sentence" )
tf.flags.DEFINE_string( "filePath" , "./../RNNdemo/CNN_inputformat_206_3000_0.2_training" , "the input data file path" )

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 202 , "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.9, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 80, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

#parse the hypeparameters
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# for attr , value in sorted( FLAGS.__flags.items() ) :
# 	print "{}={}".format( attr.upper() , value )
# print ""

"""
# data preparation
#=====================================
"""
x_train, y_train, x_dev, y_dev, embedding_mat = data_help2.load_data2(FLAGS.dataPath, FLAGS.stopwordsPath, FLAGS.w2vModelPath, 0.2, FLAGS.sequence_length)

# print "train data size is {:d} , and train lable is {:d}".format( len( x_train ) , len( y_train ) )
# print "valid data size is {:d} , and valid lable is {:d}".format( len( x_dev ) , len( y_dev ) )

"""
# Training
#=================================================
"""

with tf.Graph().as_default() :
	sess = tf.Session()
	with sess.as_default():
		cnn = TextCNN(   WW = embedding_mat ,\
						 sequence_length = FLAGS.sequence_length ,\
						 num_classes = y_train.shape[1] ,\
						 embedding_size = FLAGS.embedding_dim , \
						 filter_sizes = list( map( int , FLAGS.filter_sizes.split( "," ) ) ) , \
						 num_filters = FLAGS.num_filters , \
						 l2_reg_lambda = FLAGS.l2_reg_lambda )
	
	#define the Training produce
	global_step = tf.Variable(0 , name = "global_step" , trainable = False)
	optimizer = tf.train.AdamOptimizer( 1e-3 )
	grads_and_var = optimizer.compute_gradients( cnn.loss )
	train_op = optimizer.apply_gradients( grads_and_var , global_step = global_step )

	#output directory for models and summaries
	timestamp = str( int( time.time() ) )
	out_dir = os.path.abspath( os.path.join( os.path.curdir , "runs" , timestamp ) )
	print "Writing to {} \n".format( out_dir )

	#Summary for loss and accuracy
	loss_summary = tf.scalar_summary( "loss" , cnn.loss )
	acc_summary = tf.scalar_summary( "accuracy" , cnn.accuracy )

	#Train Summaries
	train_summary_op = tf.merge_summary([ loss_summary , acc_summary ])
	train_summary_dir = os.path.join( out_dir , "summary" , "train" )
	train_summary_writer = tf.train.SummaryWriter( train_summary_dir , sess.graph )

	#dev summaries
	dev_summary_op = tf.merge_summary( [ loss_summary , acc_summary ] )
	dev_summary_dir = os.path.join( out_dir , "summaries" , "dev" )
	dev_summary_writer = tf.train.SummaryWriter( dev_summary_dir , sess.graph )

	#create the checkpoint directory ,if not ,we create it
	checkpoint_dir = os.path.abspath( os.path.join(out_dir , "checkpoints" ) )
	checkpoint_prefix = os.path.join( checkpoint_dir , "model" )
	if not os.path.exists( checkpoint_dir ) :
		os.makedirs( checkpoint_dir )
	saver = tf.train.Saver( tf.all_variables() )


	#Initialize all variables 
	sess.run( tf.initialize_all_variables() )

	#training the step 
	def train_step( x_batch , y_batch ) :
		"""
		This is the one step for training
		"""
		feed_dict = { cnn.input_x : x_batch , cnn.input_y : y_batch , cnn.dropout_keep_prob : FLAGS.dropout_keep_prob }
		_ , step , summaries , loss , accuracy = sess.run( [ train_op , global_step , train_summary_op , cnn.loss , cnn.accuracy ] , feed_dict )
		time_str = datetime.datetime.now().isoformat()
		# print "{} : step {} , loss {:g} , acc {:g}".format( time_str , step , loss , accuracy )
		train_summary_writer.add_summary( summaries , step )

	def dev_step( x_batch , y_batch , writer = None,output=False ) :
		"""
		Evaluates the model on a dev datasets
		"""
		feed_dict = { cnn.input_x : x_batch , cnn.input_y : y_batch , cnn.dropout_keep_prob : 1.0 }
		if output :
			step,summaries,loss,accuracy,predictions,scores=sess.run([ global_step,dev_summary_op,cnn.loss,cnn.accuracy,cnn.predictions,cnn.scores],feed_dict)
		else :
			step,summaries,loss,accuracy,predictions=sess.run([ global_step,dev_summary_op,cnn.loss,cnn.accuracy,cnn.predictions],feed_dict)
		time_str = datetime.datetime.now().isoformat()
		if output :
			fw = open("RNN_CNN_SVM_input_training_both" , 'a')
			for i in range(len(y_batch)):
				vv = map(str , list(scores[i]))
				vv.append(str(y_batch[i][1]))
				vv.append("\n")
				fw.write("\t".join(vv))
			fw.close()

		predictionsList = list(predictions)
		label_y =[ ll[1] for ll in list(y_batch) ]
		if len(predictionsList) != len(label_y) :
			print "error !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

		TP = [ i for i in range(len(predictionsList)) if predictionsList[i] == 1 and label_y[i] == 1]
		TN = [ i for i in range(len(predictionsList)) if predictionsList[i] == 0 and label_y[i] == 1]
		FP = [ i for i in range(len(predictionsList)) if predictionsList[i] == 1 and label_y[i] == 0]

		print "acc {} , recall {} ".format(len(TP) *1.0/(len(TP)+len(FP)) , len(TP)*1.0/(len(TP)+len(TN)))
		print "{} : step {} , loss {:g} , acc {:g}".format( time_str , step , loss , accuracy )
		if writer :
			writer.add_summary( summaries , step )

	# generate the batch of datas
	batches = data_help2.batch_iter(list(zip(x_train, y_train ))  , FLAGS.batch_size , FLAGS.num_epochs , shuffle = True )

	for batch in batches :
		x_batch , y_batch = zip( *batch )
		# print "The length of batch x is : {:d} , y is {:d}".format(len(x_batch) , len(y_batch))
		train_step( x_batch  , y_batch  )
		current_step = tf.train.global_step( sess , global_step )
		if current_step % FLAGS.evaluate_every == 0 :
			print " Evaluation : "
			dev_step(x_dev , y_dev , writer = dev_summary_writer)
			# print ""
		if current_step % FLAGS.checkpoint_every == 0 :
			path = saver.save( sess , checkpoint_prefix , global_step = current_step )
			print "Saved model checkpoint to {}".format( path )
	# dev_step(x_data , y_data , writer = dev_summary_writer ,output=True)