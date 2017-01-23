#encoding="utf-8"
#!/usr/bin/env python

import sys
import numpy as np
import tensorflow as tf
import time

import data_help2

reload( sys )
sys.setdefaultencoding( "utf-8" )

#input parameters
tf.flags.DEFINE_string( "dataPath" , "./data/travelogueData" , "the data Path" )
tf.flags.DEFINE_string( "stopwordsPath" , "./data/stopwords.txt" , "The stopwwords path" )
tf.flags.DEFINE_string( "w2vModelPath" , "./data/w2vmodel.bin" , "the model of word2vec" )
tf.flags.DEFINE_float( "dev_sample_percentage" , 0.2 , "percentage of the train data to use for test" )
tf.flags.DEFINE_integer("sequence_length" , 35 , "The value of the input data length")
 
# LSTM parameters
tf.flags.DEFINE_string("model", "train","A type of model. Possible options are: small, medium, large.")
tf.flags.DEFINE_string( "save_path" , "./resultSaver" , "Model output directory" )
tf.flags.DEFINE_bool( "use_fp16" , False , "Train using 16-bit floats instead of 32 bit floats" )

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

"""
# data preparation
#=====================================
"""
x_train, y_train, x_dev, y_dev, embedding_mat = data_help2.load_data2(FLAGS.dataPath, FLAGS.stopwordsPath, FLAGS.w2vModelPath,0.2, FLAGS.sequence_length)

train_data = list(zip(x_train , y_train))
dev_data = list(zip(x_dev , y_dev))

# total_data = list(zip(x_data , y_data))

def data_type() :
	return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBModel( object ) :
	"""
	# The model of PTB ...
	"""
	def __init__(self, is_training, config,embedding_mat):
		size = config.hidden_size
		self.input_x = tf.placeholder(tf.float32,[None,size ], name="self.input_x")
		self.input_y = tf.placeholder(tf.float32,[None,config.num_classes], name="input_y")

		# remember the init of forget gate biases for better results
		# state_is_tuple means that the Ct and Ht are tuple that store

		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size , forget_bias = 0.0 , state_is_tuple = True )
		if is_training and config.keep_prob < 1 :
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell , output_keep_prob = config.keep_prob )

		cell = tf.nn.rnn_cell.MultiRNNCell( [ lstm_cell ] * config.num_layers , state_is_tuple = True )
	
		self._initial_state = cell.zero_state( 1 , data_type() )

		with tf.device( "/cpu:0" ) :
			embedding_mat = tf.constant(embedding_mat, dtype = data_type() )
			inputs = tf.nn.embedding_lookup(embedding_mat , self.input_x )

		# we use the dropout methods to handle the inputs , and return the handled the data for the inputs
		if is_training and config.keep_prob < 1 :
			self.input_x = tf.nn.dropout( self.input_x , config.keep_prob )

		# forward the propagation and get the output 
		outputs = []
		temp_output = []
		state = self._initial_state
		with tf.variable_scope( "RNN" ):
			for time_step in range(config.num_steps) : 
				if time_step > 0 :
					tf.get_variable_scope().reuse_variables()
				#the raw is self.input_x[time_step,:]  which is wrong 
				#When I change to tf.reshape(self.input_x[time_step,:],[-1,size]) make true
				( cell_output , state ) = cell( tf.reshape(self.input_x[time_step , :],[-1,size] ), state )
				temp_output.append( cell_output )
				if time_step == config.num_steps - 1:
					outputs.append( cell_output )

		temp_tensor1 = tf.pack(temp_output)
		self._handleOutput = tf.reshape(temp_tensor1 , [-1])

		# change the output to standard shape
		output = tf.reshape( tf.concat( 1 , outputs ) , [ - 1 , size ] )

		softmax_W = tf.get_variable( "softmax_w" , [ size , config.num_classes ] , dtype = data_type() )
		softmax_b = tf.get_variable( "softmax_b" , [ config.num_classes ] , dtype = data_type() )

		# calculate the value of inner value of two tensor
		logits = tf.matmul( output , softmax_W ) + softmax_b 


		# calculate the loss function
	# 	loss = tf.nn.seq2seq.sequence_loss_by_example([logits],[tf.reshape(self.input_y, [-1])],[tf.ones([1 * config.num_classes], dtype=data_type())])                                                      
	# #loss = tf.nn.seq2seq.sequence_loss_by_example([logits],[tf.reshape(input_.targets, [-1])],[tf.ones([batch_size * num_steps], dtype=data_type())])                                                      
		loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.reshape(logits,[-1]),tf.reshape(self.input_y, [-1]), name=None)
		self._cost = cost = tf.reduce_sum( loss )
		self._final_state = state

		self._predictions = tf.argmax(logits, 1, name="predictions")
		correct_predictions = tf.equal(self._predictions, tf.argmax(self.input_y, 1))
		self._accuracy = tf.cast(correct_predictions, "float")

		if not is_training :
			return 

		self._lr = tf.Variable( 0.0 , trainable = False )
		tvars = tf.trainable_variables()
		grads , _ = tf.clip_by_global_norm( tf.gradients( cost , tvars ) , config.max_grad_norm )

		optimizer = tf.train.GradientDescentOptimizer( self._lr )
		self._train_op = optimizer.apply_gradients( zip( grads , tvars ) , global_step = tf.contrib.framework.get_or_create_global_step() )

		self._new_lr = tf.placeholder( tf.float32 , shape = [] , name = "new_learning_rate" )
		self._lr_update = tf.assign( self._lr , self._new_lr )

	def assign_lr( self , session , lr_value ) :
		session.run( self._lr_update , feed_dict = { self._new_lr : lr_value } )

	@property
	def input( self ) :
		return self._input

	@property
	def initial_state( self ) :
		return self._initial_state 

	@property
	def cost( self ) :
		return self._cost

	@property
	def final_state( self ) :
		return self._final_state 

	@property
	def train_op( self ) :
		return self._train_op

	@property
	def lr( self ) :
		return self._lr 
	@property
	def accuracy( self ) :
		return self._accuracy

	@property
	def handleOutput( self ) :
		return self._handleOutput
	@property
	def predictions(self) :
		return self._predictions


# set the config struction of this config
class trainConfig( object ) :
	#The parameter of the configurations
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	hidden_size = 202
	max_epoch = 4
	max_max_epoch = 13
	keep_prob = 1.0
	lr_decay = 0.5
	num_classes = 2
	train_num = 1000
	valid_num = 450
	num_steps = 25

def run_epoch(session, model, data, eval_op=None, verbose=False, valid=False ,outputdata=False):
	# begin to run the data on the given data
	start_time = time.clock()
	costs = 0.0
	iters = 0
	right = 0.0
	state = session.run( model.initial_state )

	fetches = {"cost" : model.cost , "final_state" : model.final_state , "accuracy" :model.accuracy,"predictions":model.predictions}
	if eval_op is not None :
		fetches[ "eval_op" ] = eval_op
	if outputdata :
		fetches[ "handleOutput" ] = model.handleOutput

	#generate the itered data for training
	data_size = len( data )
	print "the size of data is :{:d}".format(data_size)
	begin = time.clock()
	if outputdata :
		batches = data_help2.batch_iter(data,shuffled=False)
	else :
		batches = data_help2.batch_iter(data)
	end = time.clock()
	print "The cost of time to generate batches is : %d"%(end - begin)

	PP = []
	LABEL = []

	for step , batch in enumerate(batches) :
		x_batch , y_batch = batch[0] , batch[1]
		LABEL.append(y_batch)
		feed_dict = {}
		for i , ( c , h ) in enumerate( model.initial_state ) :
			feed_dict[ c ] = state[ i ].c
			feed_dict[ h ] = state[ i ].h
		feed_dict[ model.input_x ] = np.reshape(x_batch,[-1,config.hidden_size])
		feed_dict[ model.input_y ] = np.reshape(y_batch,[-1,config.num_classes])

		vals = session.run( fetches , feed_dict )
		cost = vals[ "cost" ]
		final_state = vals[ "final_state" ]  #previous I set to state
		predictions = vals["predictions"]
		accuracy = vals[ "accuracy"]

		PP.append(float(predictions[0]))

		if outputdata :
			handleOutput = vals[ "handleOutput" ]
			out = map(str ,handleOutput.tolist())
			target = map(str , batch[1])
			out.extend(target)
			out.append("\n")
			fw = open("CNN_inputformat_206_3000_0.2_training" , 'a' )
			fw.write( "\t".join(out) )
			fw.close()

		costs += cost
		iters += config.num_steps
		right += accuracy
	
		if verbose and step % ( data_size / 10 )  == 0 :
			print "%.3f perplexity :  %.6f speed : %.0f s , accuracy is %.3f" % ( \
			step * 1.0 / data_size , \
			np.exp( costs / iters ) , \
			time.clock() - start_time ,\
			right * 1.0 / ( step + 1)) 
	TT = [ ll[1] for ll in LABEL]

	if len(PP) != len(TT) :
		print "error !!!!!!!!!!!!!!"

	TP = [ i for i in range(len(PP)) if PP[i] == 1 and TT[i] == 1]
	print "TP=",len(TP),
	FP = [ i for i in range(len(PP)) if PP[i] == 1 and TT[i] == 0]
	print "FP=",len(FP),
	TN = [ i for i in range(len(PP)) if PP[i] == 0 and TT[i] == 1]
	print "TN=",len(TN)
	if verbose :
		print "The train right : {} , recall:{} , acc : {}".format((len(TP)*1.0/(len(TP)+len(FP))),(len(TP)*1.0/(len(TP)+len(TN))),right * 1.0 / data_size)
	if valid :
		print "The valid right : {} , recall:{} , acc : {}".format((len(TP)*1.0/(len(TP)+len(FP)+0.01)),(len(TP)*1.0/(len(TP)+len(TN)+0.01)),right * 1.0 / data_size)

	return np.exp( costs / iters )

def get_config() :
	if FLAGS.model == "train" :
		return trainConfig()
	elif FLAGS.model == "test" :
		return testConfig() 
	else :
		raise ValueError("Invalid model: %s", FLAGS.model)

if __name__ == '__main__' :

	config = get_config()
	config.train_num = len( train_data	)
	config.valid_num = len( dev_data )

	with tf.Graph().as_default() :
		initializer = tf.random_uniform_initializer( - config.init_scale , config.init_scale )
		with tf.name_scope( "Train" ) :
			with tf.variable_scope( "Model" , reuse = None , initializer = initializer ) :
				m = PTBModel( is_training = True , config = config, embedding_mat =embedding_mat )

		with tf.name_scope( "Valid" ) :
			with tf.variable_scope( "Model" , reuse = True , initializer = initializer ) :
				mvalid = PTBModel( is_training = False , config = config,embedding_mat=embedding_mat )

		sv = tf.train.Supervisor(logdir=FLAGS.save_path )
		with sv.managed_session() as session:
			for i in range( config.max_max_epoch ) :
				lr_decay = config.lr_decay ** max( i + 1 - config.max_epoch , 0.0 )
				m.assign_lr( session , config.learning_rate * lr_decay )

				print "Epoch : %d Learning rate : %.3f" % ( i + 1 , session.run( m.lr ) )
				# valid_perplexity = run_epoch( session , mvalid,total_data,valid=True,outputdata=True)
				# print "begin to sleep"
				# time.sleep(600)
				# print "end to sleep"

				train_perplexity = run_epoch( session , m , train_data, eval_op = m.train_op, verbose = True )
				print "Epoch : %d Train perplexity : %.3f" % ( i + 1 , train_perplexity )

				valid_perplexity = run_epoch( session , mvalid,dev_data,valid=True)
				print "Epoch : %d Valid Perplexity : %.3f" % ( i + 1 , valid_perplexity )
			# valid_perplexity = run_epoch( session , mvalid,total_data,valid=True,outputdata=True)
				# print "begin to sleep"
				# time.sleep(600)
				# print "end to sleep"

			# test_perplexity = run_epoch( session , mtest )
			# print "Test Perplexity : %.3f" % test_perplexity

			# if FLAGS.save_path :
			# 	print "save model to %s ." % FLAGS.save_path
			# 	sv.saver.save( session , FLAGS.save_path , global_step = sv.global_step )

