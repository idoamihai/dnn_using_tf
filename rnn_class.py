from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os

def import_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    x_train = mnist.train.images[:1000,:]
    y_train = mnist.train.labels[:1000,:]
    x_valid = mnist.validation.images[:1000,:]
    y_valid = mnist.validation.labels[:1000,:]
    x_test = mnist.test.images[:1000,:]
    y_test = mnist.test.labels[:1000,:]
    # Network Parameters
    n_input = 28 # MNIST data input (img shape: 28*28)
    n_steps = 28 # timesteps
    return x_train,y_train,x_valid,y_valid,x_test,y_test,n_input,n_steps

def vectorize_labels(y):
    n_classes = len(set(y))
    y_vect = np.zeros([len(y),n_classes])
    y_vect[np.arange(len(y)), y] = 1
    return y_vect


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

    
def RNN(x, weights, biases, dropout_keep_rate, stacked_layers, n_hidden, n_steps, n_input, is_train):            
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


def BiRNN(x, weights, biases, dropout_keep_rate, stacked_layers, n_hidden, n_steps, n_input, is_train):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps, 0)
    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    if is_train:
        lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=dropout_keep_rate)
        lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=dropout_keep_rate)
    stacked_fw_lstm = rnn.MultiRNNCell([lstm_fw_cell] * stacked_layers, state_is_tuple = True)
    stacked_bw_lstm = rnn.MultiRNNCell([lstm_bw_cell] * stacked_layers, state_is_tuple = True)

    # Get lstm cell output
    outputs, _, _ = rnn.static_bidirectional_rnn(stacked_fw_lstm, stacked_bw_lstm, x,
                                          dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    if is_train:
        pred = tf.matmul(outputs[-1], weights['out']) + biases['out'] 
    else:
        pred = tf.matmul(outputs[-1], weights['out']*dropout_keep_rate) + biases['out']  
    return pred, weights['out']        

class RNN_model:
    
    def __init__(self):
        self.learning_rate_init = 0.001
        self.training_iters = 10
        self.batch_size = 12
        self.display_step = 1
        self.stacked_layers = 1
        self.patience_initial = 0.0
        self.patience = 30
        self.improvement_min_perc = 0.0
        self.dropout_keep_rate = 1.0
        self.l2_reg = 0.0
        self.opt_function = tf.train.AdamOptimizer()
        self.bidirectional = True
        self.exponential = False
        self.gradient_clipping = True
        self.n_hidden = 128
        self.path = os.getcwd()
        
    def dump(self):
        return dict(self.__dict__)
    
    def run_mnist(self):
        x_train,y_train,x_valid,y_valid,x_test,y_test,n_input,n_steps = import_mnist()
        mnist_preds = RNN_model.build_model(self,x_train,y_train,
                                            n_input,n_steps,is_train=True,run_test=True,
                                            x_valid=x_valid,y_valid=y_valid,
                                            x_test=x_test,y_test=y_test)
        return mnist_preds
    
    def build_model(self,x_train,y_train,n_input,n_steps,is_train,run_test,**kwargs):
            params = RNN_model.dump(self)
            params.update(kwargs)  
            if len(np.shape(y_train)) == 1:
                y_train = vectorize_labels(y_train)
            n_classes = y_train.shape[1]
            graph = tf.Graph()
            with graph.as_default():
                if self.exponential:
                    global_step = tf.Variable(0, trainable=False)
                    learning_rate = tf.train.exponential_decay(
                    params['learning_rate_init'], global_step, 100, 0.96, staircase=True) 
                else:
                    learning_rate = params['learning_rate_init']
                # tf Graph input
                x = tf.placeholder("float", [None, n_steps, n_input])
                y = tf.placeholder("float", [None, n_classes])
                reg_lambda = tf.placeholder(tf.float32)
                mult = 2 if params['bidirectional'] ==True else 1                
                # Define weights
                weights = {
                    'out': tf.Variable(tf.random_normal([params['n_hidden']*mult, n_classes]))
                }
                biases = {
                    'out': tf.Variable(tf.random_normal([n_classes]))
                }
                
                if params['bidirectional']==True:
                   pred, weights = BiRNN(x, weights, biases, params['dropout_keep_rate'], 
                                         params['stacked_layers'],params['n_hidden'],
                                               n_steps, n_input, is_train) 
                else:
                    pred, weights = RNN(x, weights, biases, params['dropout_keep_rate'], 
                                         params['stacked_layers'],params['n_hidden'],
                                               n_steps, n_input, is_train) 
                # Define loss and optimizer
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) +\
                                     reg_lambda * tf.nn.l2_loss(weights)
                if params['gradient_clipping'] == True:
                    with tf.name_scope('optimizer'):
                        #opt_function = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                        #opt_function = tf.train.AdamOptimizer(learning_rate=learning_rate_init)
                        opt_function = params['opt_function']
                        opt_function._lr = learning_rate
                        gradients = opt_function.compute_gradients(cost)
                        capped_gradients = [(tf.clip_by_norm(grad, clip_norm = 5.0), var) for grad, var in gradients]
                        optimizer = opt_function.apply_gradients(capped_gradients)
                else:
                    optimizer = params['opt_function']
                    optimizer._lr = learning_rate
                # Summarize all gradients
                for grad, var in gradients:
                    tf.summary.histogram(var.name + '/gradients', grad)                
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
                if 'x_valid' and 'y_valid' in params:
                    x_valid, y_valid = params['x_valid'],params['y_valid']
                    if len(np.shape(y_valid)) == 1:
                        y_valid = vectorize_labels(y_valid)
                else:
                     x_valid, y_valid = x_train,y_train
                if 'x_test' and 'y_test' in params:
                    x_test, y_test = params['x_test'],params['y_test']
                    if len(np.shape(y_test)) == 1:
                        y_test = vectorize_labels(y_test)                    
                else:
                     x_test, y_test = x_train,y_train                   
                saver = tf.train.Saver()
                summary_writer = tf.summary.FileWriter(logdir=params['path']+'/logdir/train', graph=tf.get_default_graph())
                summary_valid = tf.summary.FileWriter(logdir=params['path']+'/logdir/validation', graph=tf.get_default_graph())
                sess.run(init)
                best_validation_loss = np.inf
                best_iteration_step = 0
                patience_steps = 0
                for iteration in range(params['training_iters']):
                    total_batches = int(len(y_train)/params['batch_size'])
                    batched_data = batch_data(x_train,y_train,params['batch_size'])
                    for batch in range(total_batches):
                        batch_x, batch_y = batched_data.next()
                        # Reshape data to get 28 seq of 28 elements
                        batch_x = batch_x.reshape((batch_x.shape[0], n_steps, n_input))
                        # Run optimization op (backprop)
                        _,summary = sess.run((optimizer,merged_summary_op), 
                                 feed_dict={x: batch_x, y: batch_y,reg_lambda:params['l2_reg']})
                        summary_writer.add_summary(summary, iteration)
                    if iteration % params['display_step'] == 0:
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
                            saver.save(sess, params['path']+'lstm_best.ckpt')
                            if vloss < best_validation_loss*params['improvement_min_perc']:
                                patience_steps = 0
                            else:
                                patience_steps += 1 #small improvement
                            best_validation_loss = vloss
                            best_iteration_step = iteration
                        else:
                            patience_steps += 1
                        if ((patience_steps >= params['patience_initial']) and (iteration >= params['patience'])):
                            saver.restore(sess, params['path']+'lstm_best.ckpt')
                            test_data = x_test.reshape((-1,n_steps,n_input))
                            print("Testing Accuracy:", \
                                  sess.run(accuracy, feed_dict={x: test_data, y: y_test}))
                            print ('saved model iteration %s' %str(best_iteration_step))
                            break                                
                if run_test==True:
                    test_data = x_test.reshape((x_test.shape[0], n_steps, n_input))
                    test_label = y_test
                    predictions = sess.run(pred, feed_dict={x: test_data, y: test_label})
                    return predictions
                print("Optimization Finished!")
            


