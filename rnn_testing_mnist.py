from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''
path = os.getcwd()

x_train = mnist.train.images[:10000,:]
y_train = mnist.train.labels[:10000,:]
x_valid = mnist.validation.images[:10000,:]
y_valid = mnist.validation.labels[:10000,:]
x_test = mnist.test.images[:10000,:]
y_test = mnist.test.labels[:10000,:]


# Parameters
learning_rate_init = 0.001
training_iters = 20
batch_size = 128
display_step = 1

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

stacked_layers = 1
patience_initial = 0.0
patience = 30
improvement_min_perc = 0.0
dropout_keep_rate = 1.0
l2_reg = 0.0

is_train=True
run_test=True



def batch_data(x,y,batch_size):
    idx = range(len(y))
    np.random.shuffle(idx)
    x,y = x[idx],y[idx]        
    total_batches = int(len(y)/batch_size)
    batch_x = np.array_split(x,total_batches)
    batch_y = np.array_split(y,total_batches)
    batched_data = [(i,j) for i,j in zip(batch_x,batch_y)]
    batched_data = iter(batched_data) 
    return batched_data

graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
    learning_rate_init, global_step, 100, 0.96, staircase=True)    
    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    reg_lambda = tf.placeholder(tf.float32)
    
    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    
    def RNN(x, weights, biases):            
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)            
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, n_steps, 0)            
        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        if is_train:
            lstm_cell = rnn.DropoutWrapper(lstm_cell,output_keep_prob=dropout_keep_rate)
        stacked_lstm = rnn.MultiRNNCell([lstm_cell] * stacked_layers,
                                             state_is_tuple=True)
        # Get lstm cell output
        outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)            
        # Linear activation, using rnn inner loop last output
        if is_train:
            pred = tf.matmul(outputs[-1], weights['out']) + biases['out'] 
        else:
            pred = tf.matmul(outputs[-1], weights['out']*dropout_keep_rate) + biases['out']  
        return pred, weights['out']        
    
    pred, weights = RNN(x, weights, biases)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) +\
                         reg_lambda * tf.nn.l2_loss(weights)
    with tf.name_scope('optimizer'):
        #opt_function = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        opt_function = tf.train.AdamOptimizer(learning_rate=learning_rate_init)
        gradients = opt_function.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_norm(grad, clip_norm = 5.0), var) for grad, var in gradients]
        optimizer = opt_function.apply_gradients(capped_gradients)
    # Summarize all gradients
    for grad, var in gradients:
        tf.summary.histogram(var.name + '/gradient', grad)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    #save some variables
    tf.summary.scalar("cost_function", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all() 

# Launch the graph
with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(logdir=path+'/logdir/train', graph=tf.get_default_graph())
    summary_valid = tf.summary.FileWriter(logdir=path+'/logdir/validation', graph=tf.get_default_graph())
    sess.run(init)
    best_validation_loss = np.inf
    best_step = 0
    patience_steps = 0
    tracking = []
    for iteration in range(training_iters):
        total_batches = int(len(y_train)/batch_size)
        batched_data = batch_data(x_train,y_train,batch_size)
        for batch in range(total_batches):
            batch_x, batch_y = batched_data.next()
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_x.shape[0], n_steps, n_input))
            # Run optimization op (backprop)
            _,summary = sess.run((optimizer,merged_summary_op), 
                     feed_dict={x: batch_x, y: batch_y,reg_lambda:l2_reg})
            summary_writer.add_summary(summary, iteration)
        if iteration % display_step == 0:
            # Calculate batch accuracy
            x_val = x_valid.reshape((x_valid.shape[0],n_steps,n_input))
            vacc,vloss,vsummary = sess.run((accuracy,cost,merged_summary_op), 
                                  feed_dict={x: x_val, y: y_valid,
                                                reg_lambda:0.0})
            summary_valid.add_summary(vsummary,iteration)
            print("Iter %d" %(iteration)+ ", Minibatch Loss= " + \
                  "{:.6f}".format(vloss) + ", Training Accuracy= " + \
                  "{:.5f}".format(vacc))            
            if vloss < best_validation_loss:
                saver.save(sess, path+'lstm_best.ckpt')
                if vloss < best_validation_loss*improvement_min_perc:
                    patience_steps = 0
                else:
                    patience_steps += 1 #small improvement
                best_validation_loss = vloss
                best_iteration_step = iteration
            else:
                patience_steps += 1
            if ((patience_steps >= patience_initial) and (iteration >= patience)):
                saver.restore(sess, path+'lstm_best.ckpt')
                test_data = x_test.reshape((-1,n_timesteps,n_input))
                print("Testing Accuracy:", \
                      sess.run(accuracy, feed_dict={x: test_data, y: y_test}))
                print ('saved model iteration %s' %str(best_iteration_step))
                break                                
    if run_test==True:
        test_data = x_test.reshape((x_test.shape[0], n_steps, n_input))
        test_label = y_test
        predictions = sess.run(pred, feed_dict={x: test_data, y: test_label})
    print("Optimization Finished!")


