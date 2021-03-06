from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os

'''
fix validation predictions
'''


is_train=True
params = {'learning_rate_init':0.001,
        'training_iters':10,
        'batch_size':10,
        'display_step':1,
        'stacked_layers':1,
        'patience_initial':0,
        'patience':30,
        'improvement_min_perc':0.0,
        'dropout_keep_rate':0.5,
        'l2_reg':0.001,
        'l2_reg_dense':0.01,
        'opt_function':tf.train.AdamOptimizer(),
        'bidirectional':True,
        'exponential':False,
        'gradient_clipping':True,
        'n_hidden':128,
        'path':os.getcwd(),
        'n_hidden_1_dense':256,
        'n_hidden_2_dense':256}

def import_mnist(limit=1000):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    x_train = mnist.train.images[:limit,:]
    y_train = mnist.train.labels[:limit,:]
    x_valid = mnist.validation.images[:limit,:]
    y_valid = mnist.validation.labels[:limit,:]
    x_test = mnist.test.images[:limit,:]
    y_test = mnist.test.labels[:limit,:]
    # Network Parameters
    n_input = 28 # MNIST data input (img shape: 28*28)
    n_steps = 28 # timesteps
    return x_train,y_train,x_valid,y_valid,x_test,y_test,n_input,n_steps

def vectorize_labels(y):
    n_classes = len(set(y))
    y_vect = np.zeros([len(y),n_classes])
    y_vect[np.arange(len(y)), y.astype(int)] = 1
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
    
def RNN(x, weights, biases, dropout_keep_rate, stacked_layers, 
        n_hidden, n_steps, n_input, is_train):            
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
    return pred 

def BiRNN(x, weights, biases, dropout_keep_rate, stacked_layers, 
          n_hidden, n_steps, n_input, is_train):
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
    return pred

def dense_nn(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

class RNN_model:
    
    def __init__(self):
        self.learning_rate_init = 0.001
        self.training_iters = 100
        self.batch_size = 100
        self.display_step = 1
        self.stacked_layers = 1
        self.patience_initial = 0
        self.patience = 30
        self.improvement_min_perc = 0.0
        self.dropout_keep_rate = 1.0
        self.l2_reg = 0.0
        self.l2_reg_dense = 0.01
        self.opt_function = tf.train.AdamOptimizer()
        self.bidirectional = True
        self.exponential = False
        self.gradient_clipping = True
        self.n_hidden = 100
        self.path = os.getcwd()
        self.n_hidden_1_dense = 100
        self.n_hidden_2_dense = 100
        
    def dump(self):
        return dict(self.__dict__)
    
    def run_mnist(self):
        x_train,y_train,x_valid,y_valid,x_test,y_test,n_input,n_steps = import_mnist()
        x_aux_train = x_train
        x_aux_valid = x_valid
        x_aux_test = x_test
        mnist_preds = RNN_model.build_model(self,x_train,y_train,x_aux_train,
                                            n_input,n_steps,run_test=True,
                                            x_valid=x_valid,y_valid=y_valid,x_aux_valid=x_aux_valid,
                                            x_test=x_test,y_test=y_test,x_aux_test=x_aux_test,
                                            train_test='train')
        return mnist_preds
    
    def build_model(self,x_train,y_train,x_aux_train,n_input,n_steps,run_test,train_test,**kwargs):
            params = RNN_model.dump(self)
            params.update(kwargs)  
            if len(np.shape(y_train)) == 1:
                y_train = vectorize_labels(y_train)
            n_classes = y_train.shape[1]
            n_input_dense = x_aux_train.shape[1]
            graph = tf.Graph()
            with graph.as_default():
                '''
                if self.exponential:
                    global_step = tf.Variable(0, trainable=False)
                    learning_rate = tf.train.exponential_decay(
                    params['learning_rate_init'], global_step, 100, 0.96, staircase=True) 
                else:
                    learning_rate = params['learning_rate_init']
                '''
                learning_rate = params['learning_rate_init']
                # tf Graph input
                x = tf.placeholder("float", [None, n_steps, n_input])
                y = tf.placeholder("float", [None, n_classes])
                x_dense = tf.placeholder("float", [None,  n_input_dense])
                x_validation = tf.placeholder("float", [None, n_steps, n_input])
                y_validation = tf.placeholder("float", [None, n_classes])
                x_validation_dense = tf.placeholder("float", [None,  n_input_dense])
                x_testing = tf.placeholder("float", [None, n_steps, n_input])
                y_testing = tf.placeholder("float", [None, n_classes])
                x_testing_dense = tf.placeholder("float", [None,  n_input_dense])                
                
                reg_l2 = tf.placeholder(tf.float32)
                reg_l2_dense = tf.placeholder(tf.float32)
                mult = 2 if params['bidirectional'] ==True else 1                
                
                # Define weights
                glorot_lim = tf.sqrt(6.0 / (n_input*n_steps + n_classes))
                weights_rnn = {
                    'out': tf.Variable(tf.random_uniform([params['n_hidden']*mult, n_classes],
                                                         minval=-glorot_lim,maxval=glorot_lim))
                }
                biases_rnn = {
                    'out': tf.Variable(tf.zeros([n_classes]))
                }
                
                weights_dense = {
                    'h1': tf.Variable(tf.random_normal([n_input_dense, params['n_hidden_1_dense']])),
                    'h2': tf.Variable(tf.random_normal([params['n_hidden_1_dense'], params['n_hidden_2_dense']])),
                    'out': tf.Variable(tf.random_normal([params['n_hidden_2_dense'], n_classes]))
                }
                biases_dense = {
                    'b1': tf.Variable(tf.random_normal([params['n_hidden_1_dense']])),
                    'b2': tf.Variable(tf.random_normal([params['n_hidden_2_dense']])),
                    'out': tf.Variable(tf.random_normal([n_classes]))
                }                
                
                
                if params['bidirectional']==True:
                   pred_rnn = BiRNN(x, weights_rnn, biases_rnn, params['dropout_keep_rate'], 
                                         params['stacked_layers'],params['n_hidden'],
                                               n_steps, n_input, is_train=True)
                else:
                    pred_rnn = RNN(x, weights_rnn, biases_rnn, params['dropout_keep_rate'], 
                                         params['stacked_layers'],params['n_hidden'],
                                               n_steps, n_input, is_train=True) 
                pred_dense = dense_nn(x_dense, weights_dense, biases_dense)
                
                all_logits = tf.concat([pred_rnn,pred_dense],axis=1)
                
                weights_output = tf.Variable(tf.random_normal([
                        n_classes*2, n_classes]))
    
                biases_output = tf.Variable(tf.random_normal([n_classes]))
                
                all_logits = tf.matmul(all_logits,weights_output)+biases_output 
                                    
                # Define loss and optimizer
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=all_logits, labels=y)) #\
                    #+ reg_l2 * tf.nn.l2_loss(pred_rnn) + reg_l2_dense * tf.nn.l2_loss(pred_dense)
                    
                all_preds = tf.nn.softmax(all_logits)
                
                with tf.variable_scope('validation'):
                    if params['bidirectional']==True:
                        pred_rnn_valid = BiRNN(x_validation, weights_rnn, biases_rnn, params['dropout_keep_rate'], 
                                             params['stacked_layers'],params['n_hidden'],
                                                   n_steps, n_input, is_train=False)
                    else:
                        pred_rnn_valid = RNN(x_validation, weights_rnn, biases_rnn, params['dropout_keep_rate'], 
                                             params['stacked_layers'],params['n_hidden'],
                                                   n_steps, n_input, is_train=False) 
                    pred_dense_valid = dense_nn(x_validation_dense, weights_dense, biases_dense)                                 
                    all_preds_valid = tf.concat([pred_rnn_valid,pred_dense_valid],axis=1)
                    #validation predictions
                    all_preds_valid = tf.nn.softmax(tf.matmul(all_preds_valid,weights_output)+biases_output)
                    cost_valid = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=all_preds_valid, labels=y_validation)) 

                with tf.variable_scope('test'):
                    if params['bidirectional']==True:
                        pred_rnn_test = BiRNN(x_testing, weights_rnn, biases_rnn, params['dropout_keep_rate'], 
                                             params['stacked_layers'],params['n_hidden'],
                                                   n_steps, n_input, is_train=False)
                    else:
                        pred_rnn_test = RNN(x_testing, weights_rnn, biases_rnn, params['dropout_keep_rate'], 
                                             params['stacked_layers'],params['n_hidden'],
                                                   n_steps, n_input, is_train=False) 
                    pred_dense_test = dense_nn(x_testing_dense, weights_dense, biases_dense)                                 
                    all_preds_test = tf.concat([pred_rnn_test,pred_dense_test],axis=1)
                    #validation predictions
                    all_preds_test = tf.nn.softmax(tf.matmul(all_preds_test,weights_output)+biases_output)
                    cost_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=all_preds_test, labels=y_testing)) 

                
                if params['gradient_clipping'] == True:
                    with tf.name_scope('optimizer'):
                        #opt_function = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                        #opt_function = tf.train.AdamOptimizer(learning_rate=learning_rate_init)
                        opt_function = params['opt_function']
                        opt_function._lr = learning_rate
                        gradients = opt_function.compute_gradients(cost)
                        gradients_ = [g for g in gradients if g[0]!=None]
                        capped_gradients = [(tf.clip_by_norm(grad, clip_norm = 5.0), var) for grad, var in gradients_]
                        optimizer = opt_function.apply_gradients(capped_gradients)
                else:
                    optimizer = params['opt_function']
                    optimizer._lr = learning_rate
                # Summarize all gradients
                gradients_ = [g for g in gradients if g[0]!=None]
                for grad, var in gradients_:
                    tf.summary.histogram(var.name + '/gradients', grad)                
                # Evaluate model
                correct_pred = tf.equal(tf.argmax(all_preds,1), tf.argmax(y,1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
                correct_pred_valid = tf.equal(tf.argmax(all_preds_valid,1), tf.argmax(y_validation,1))
                accuracy_valid = tf.reduce_mean(tf.cast(correct_pred_valid, tf.float32))
                correct_pred_test = tf.equal(tf.argmax(all_preds_test,1), tf.argmax(y_testing,1))
                accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))                
                # Initializing the variables
                init = tf.global_variables_initializer()
                #save some variables
                tf.summary.scalar("cost_function", cost)
                # Create a summary to monitor accuracy tensor
                tf.summary.scalar("accuracy", accuracy)
                #validation summaries
                tf.summary.scalar("validation_loss", cost_valid)
                tf.summary.scalar("validation_accuracy", accuracy_valid)
                #test summaries
                tf.summary.scalar("validation_loss", cost_test)
                tf.summary.scalar("validation_accuracy", accuracy_test)                               
                # Merge all summaries into a single op
                merged_summary_op = tf.summary.merge_all() 

            # Launch the graph
            with tf.Session(graph=graph) as sess:
                saver = tf.train.Saver()
                if train_test=='train':
                    if 'x_valid' and 'y_valid' in params:
                        x_valid, y_valid = params['x_valid'],params['y_valid']
                        x_aux_valid = params['x_aux_valid']
                        if len(np.shape(y_valid)) == 1:
                            y_valid = vectorize_labels(y_valid)
                    else:
                         x_valid, y_valid, x_aux_valid = x_train,y_train,x_aux_train
                    if 'x_test' and 'y_test' in params:
                        x_test, y_test = params['x_test'],params['y_test']
                        x_aux_test = params['x_aux_test']
                        if len(np.shape(y_test)) == 1:
                            y_test = vectorize_labels(y_test)                    
                    else:
                         x_test, y_test, x_aux_test = x_train,y_train,x_aux_train                   
                    summary_writer = tf.summary.FileWriter(logdir=params['path']+'/logdir/train', graph=tf.get_default_graph())
                    sess.run(init)
                    best_validation_loss = np.inf
                    best_step = 0
                    patience_steps = 0
                    step = 0
                    for iteration in range(params['training_iters']):
                        total_batches = int(len(y_train)/params['batch_size'])
                        batched_data = batch_data(x_train,y_train,params['batch_size'])
                        batched_aux_data = batch_data(x_aux_train,y_train,params['batch_size'])
                        for batch in range(total_batches):
                            step += 1
                            batch_x, batch_y = batched_data.next()
                            # Reshape data to get 28 seq of 28 elements
                            batch_x_aux,_ = batched_aux_data.next()
                            batch_x = batch_x.reshape((batch_x.shape[0], n_steps, n_input))
                            x_val = x_valid.reshape((x_valid.shape[0],n_steps,n_input))
                            x_tes = x_test.reshape((x_test.shape[0],n_steps,n_input))
                            # Run optimization op (backprop)
                            _,acc,loss,summary,vloss,vacc,tloss,tacc = sess.run((optimizer,accuracy,
                                     cost,merged_summary_op,cost_valid,accuracy_valid,
                                     cost_test,accuracy_test), 
                                     feed_dict={x: batch_x, y: batch_y, x_dense: batch_x_aux,
                                                x_validation: x_val, y_validation: y_valid,
                                                x_validation_dense: x_aux_valid,
                                                x_testing: x_tes, y_testing: y_test,
                                                x_testing_dense: x_aux_test,
                                                reg_l2:params['l2_reg'],reg_l2_dense:params['l2_reg_dense']})
                            summary_writer.add_summary(summary, iteration)
                            if step % params['display_step'] == 0:
                                print("Iter %d batch %d" %(iteration,batch)+ ", Minibatch Loss= " + \
                                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                                      "{:.5f}".format(acc) + ", Validation Accuracy= " + \
                                       "{:.5f}".format(vacc) +", Validataion Loss = " + \
                                        "{:.6f}".format(vloss))            
                                if vloss < best_validation_loss:
                                    saver.save(sess, params['path']+'lstm_best.ckpt')
                                    if vloss < best_validation_loss*params['improvement_min_perc']:
                                        patience_steps = 0
                                    else:
                                        patience_steps += 1 #small improvement
                                    best_validation_loss = vloss
                                    best_step = step
                                else:
                                    patience_steps += 1
                                if ((patience_steps >= params['patience_initial']) and (step >= params['patience'])):
                                    print("Testing Accuracy:", tacc)
                                    print ('saved model iteration %s' %str(best_step))
                                    break                                
                    print("Testing Accuracy:", tacc)
                    print ('saved model iteration %s' %str(best_step))
                    print("Optimization Finished!")
                elif train_test=='test':
                    x_test,y_test,x_aux_test = x_train,y_train,x_aux_train
                    saver.restore(sess, params['path']+'lstm_best.ckpt')
                    test_data = x_test.reshape((-1,n_steps,n_input))
                    test_label = y_test
                    predictions = sess.run(all_preds, feed_dict={x: test_data, y: test_label,
                                                     x_dense: x_aux_test,reg_l2:0.0,reg_l2_dense:0.0})
                    return predictions                    
            


