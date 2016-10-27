# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:32:35 2016

@author: idoamihai
"""

import tensorflow as tf
import numpy as np, pandas as pd
from sklearn import model_selection, preprocessing, metrics
import random
from tensorflow.python.ops import rnn, rnn_cell


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def grid_search_lstm(param_grid,x_train,y_train,num_labels=2,n_input=4,n_timesteps=31,n_folds=5,scoring=metrics.accuracy_score):
    max_score = 0
    max_params = {}
    grid = model_selection.ParameterGrid(param_grid)
    grid_search_scores = []
    for params_ in grid:
        clf = lstm(params_)
        score,steps = clf.cv_score(x_train,y_train,n_input,n_timesteps,num_labels,n_folds,scoring=scoring,predict_proba=False)
        if score > max_score:
            max_score = score
            max_params = params_
        dict_ = params_
        dict_['score'] = score
        dict_['steps'] = steps
        grid_search_scores.append(dict_) 
    return grid_search_scores,max_params,max_score     

  
class lstm():        
    def __init__(self,params,**kwargs):
        self.params = params
        if 'path' in kwargs: #path for saving and restoring models
            self.path = kwargs['path']
        else:
            self.path = ''

    def fit_predict(self,train_data,train_labels,n_input,n_timesteps,n_classes,is_train,predict_proba=False,
                    clipping=False,bidirectional=True,**kwargs):
        graph = tf.Graph()
        with graph.as_default():
            # Parameters
            if 'scale' in self.params:
                scaler = preprocessing.StandardScaler()
                scaler.fit(train_data)        
                train_data = scaler.transform(train_data)
                if 'x_valid' and 'y_valid' in kwargs:
                    kwargs['x_valid'] = scaler.transform(kwargs['x_valid'])
                if 'x_test' and 'y_test' in kwargs:
                    kwargs['test_data'] = scaler.transform(kwargs['test_data'])              
            if self.params['learning_rate'] == 'exp_decay':
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(
               0.001, global_step, 100, 0.96, staircase=True) 
            else:                
                learning_rate = self.params['learning_rate']
            training_iters = self.params['training_iters']
            if 'patience' in self.params:
                patience = self.params['patience']
                patience_increase = 5 #wait this much longer when a new best is found
                improvement_threshold = 0.995 # a relative improvement of this much is considered significant
            if 'random_seed' in self.params:
                random.seed(self.params['random_seed'])
            if 'batch_size' in self.params:
                batch_size = self.params['batch_size']
            else:
                batch_size = len(train_data) #batch optimizer
            if 'n_layers' in self.params:
                n_layers = self.params['n_layers']
            else:
                n_layers = 1
            display_step = self.params['display_step']            
            # Network Parameters
            n_hidden = self.params['n_hidden'] # hidden layer num of features
            if is_train == False:
                keep_prob = 1.0 #no dropout
            else:
                if 'keep_prob' in self.params:
                    keep_prob = self.params['keep_prob']
                else:
                    keep_prob = 1.0
            n_classes = n_classes             
            # tf Graph input
            x = tf.placeholder("float", [None, n_timesteps, n_input])
            y = tf.placeholder("float", [None, n_classes])            
            # Define weights
            if bidirectional:
                mult = 2
            else:
                mult = 1
            weights = {
                'out': tf.Variable(tf.random_normal([mult*n_hidden, n_classes]))
            }
            biases = {
                'out': tf.Variable(tf.random_normal([n_classes]))
            }   

            beta_regul = tf.placeholder(tf.float32)
                                        
            def RNN(x, weights, biases):            
                # Prepare data shape to match `rnn` function requirements
                # Current data input shape: (batch_size, n_steps, n_input)
                # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)            
                # Permuting batch_size and n_steps
                x = tf.transpose(x, [1, 0, 2])
                # Reshaping to (n_steps*batch_size, n_input)
                x = tf.reshape(x, [-1, n_input])
                # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
                x = tf.split(0, n_timesteps, x)            
                # Define a lstm cell with tensorflow
                lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True) 
                if is_train:
                    lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)               
                                       
                stacked_lstm = rnn_cell.MultiRNNCell([lstm_cell] * n_layers,
                                                     state_is_tuple=True)
                # Get lstm cell output
                outputs, states = rnn.rnn(stacked_lstm, x, dtype=tf.float32)            
                # Linear activation, using rnn inner loop last output
                if is_train:
                    pred = tf.matmul(outputs[-1], weights['out']) + biases['out'] 
                else:
                    pred = tf.matmul(outputs[-1], weights['out']*keep_prob) + biases['out']  
                    #perform weight scaling during testing                  
                return pred, weights['out']           

            
            def BiRNN(x, weights, biases):
            
                # Prepare data shape to match `bidirectional_rnn` function requirements
                # Current data input shape: (batch_size, n_steps, n_input)
                # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
            
                # Permuting batch_size and n_steps
                x = tf.transpose(x, [1, 0, 2])
                # Reshape to (n_steps*batch_size, n_input)
                x = tf.reshape(x, [-1, n_input])
                # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
                x = tf.split(0, n_timesteps, x)
            
                # Define lstm cells with tensorflow
                # Forward direction cell
                lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0,state_is_tuple=True)
                # Backward direction cell
                lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0,state_is_tuple=True)
                if is_train:
                    lstm_fw_cell = rnn_cell.DropoutWrapper(lstm_fw_cell, 
                                       output_keep_prob=keep_prob)  
                    lstm_bw_cell = rnn_cell.DropoutWrapper(lstm_bw_cell, 
                                       output_keep_prob=keep_prob) 
                lstm_fw_cell = rnn_cell.MultiRNNCell([lstm_fw_cell]*n_layers,state_is_tuple=True)
                lstm_bw_cell = rnn_cell.MultiRNNCell([lstm_bw_cell]*n_layers,state_is_tuple=True)
            
                # Get lstm cell output
                outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                          dtype=tf.float32)
                if is_train:
                    pred = tf.matmul(outputs[-1], weights['out']) + biases['out'] 
                else:
                    pred = tf.matmul(outputs[-1], weights['out']*keep_prob) + biases['out']  
                    #perform weight scaling during testing                 
            
                # Linear activation, using rnn inner loop last output
                return pred, weights['out']
            
            if bidirectional:
                pred, output_weights = BiRNN(x, weights, biases)
            else:
                pred, output_weights = RNN(x, weights, biases) 
            
            if n_classes == 2:
                with tf.name_scope('model'):
                    probabilities = tf.nn.sigmoid(pred) #not exactly probabilities but often close enough
                with tf.name_scope('cost_function'):
                    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y)) +\
                    beta_regul * tf.nn.l2_loss(weights['out'])
            else:
                with tf.name_scope('model'):
                    probabilities = tf.nn.softmax(pred)
                with tf.name_scope('cost_function'):
                    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) +\
                    beta_regul * tf.nn.l2_loss(weights['out'])
            # Define loss and optimizer
            with tf.name_scope('optimizer'):
                opt_function = tf.train.AdamOptimizer(learning_rate=learning_rate)
                gradients = opt_function.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_norm(grad, clip_norm = 5.0), var) for grad, var in gradients]
                if clipping:
                    optimizer = opt_function.apply_gradients(capped_gradients)
                else:
                    optimizer = opt_function.apply_gradients(gradients)                    
            # Summarize all gradients
            for grad, var in gradients:
                tf.histogram_summary(var.name + '/gradient', grad)
            # Evaluate model
            correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))            
            # Initializing the variables
            init = tf.initialize_all_variables()               
            # Create a summary to monitor cost tensor
            tf.scalar_summary("cost_function", cost)
            # Create a summary to monitor accuracy tensor
            tf.scalar_summary("accuracy", accuracy)
            # Merge all summaries into a single op
            merged_summary_op = tf.merge_all_summaries()            
        # Launch the graph
        with tf.Session(graph=graph) as sess:
            sess.run(init)
            saver = tf.train.Saver()
            # op to write logs to Tensorboard
            summary_writer = tf.train.SummaryWriter(logdir=self.path+'/logdir', graph=tf.get_default_graph())
            # Keep training until reach max iterations
            if is_train:
                best_validation_loss = np.inf
                best_training_loss = np.inf
                best_step = 0
                patience_steps = 0
                tracking = []
                for step in range(training_iters):
                    total_batches = int(train_data.shape[0]/batch_size)
                    batched_data = np.array_split(np.c_[train_data,train_labels],total_batches)
                    #shuffle
                    idx = range(len(batched_data))
                    np.random.shuffle(idx)
                    batched_data = np.array(batched_data)[idx]
                    avg_cost = 0.
                    for batch in range(total_batches):
                        #offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                        #batch_x = train_data[offset:(offset + batch_size), :]
                        #batch_y = train_labels[offset:(offset + batch_size), :]
                        batch_x = batched_data[batch][:,:-n_classes]
                        batch_y = batched_data[batch][:,-n_classes:]
                        batch_x = batch_x.reshape((batch_x.shape[0], n_timesteps, n_input))
                        # Run optimization op (backprop)
                        if 'l2_regul' in self.params:
                            regul = self.params['l2_regul']
                        else:
                            regul = 0.0
                        _, loss, ow, summary = sess.run([optimizer, cost,
                                output_weights, merged_summary_op],feed_dict={x: batch_x, y: batch_y, beta_regul: regul})
                        # Write logs at every iteration
                        summary_writer.add_summary(summary, step * total_batches + batch)
                        # Compute average loss
                        avg_cost += loss / total_batches
                    if step % display_step == 0:  
                        if 'x_valid' and 'y_valid' in kwargs:
                            x_valid = kwargs['x_valid'].reshape((-1,n_timesteps,n_input))
                            vl, summary_ = sess.run([cost,merged_summary_op],feed_dict={x:x_valid,y:kwargs['y_valid'], beta_regul: 0})
                            summary_writer.add_summary(summary_, step)
                            tracking.append([step,loss,vl,ow])                  
                            if 'patience' in self.params:
                                if vl < best_validation_loss:
                                    saver.save(sess, self.path+'lstm_best.ckpt')
                                    if vl < best_validation_loss*improvement_threshold:
                                        patience_steps = 0
                                    else:
                                        patience_steps += 1 #small improvement
                                    best_validation_loss = vl
                                    best_step = step
                                else:
                                    patience_steps += 1
                                if ((patience_steps >= patience_increase) and (step >= patience)):
                                    if 'x_test' and 'y_test' in kwargs:
                                        saver.restore(sess, self.path+'lstm_best.ckpt')
                                        test_data = kwargs['test_data'].reshape((-1,n_timesteps,n_input))
                                        print("Testing Accuracy:", \
                                              sess.run(accuracy, feed_dict={x: test_data, y: kwargs['test_labels']}))
                                    print 'saved model step %d' % (best_step)
                                    break  
                            elif vl < best_validation_loss:
                                saver.save(sess, self.path+'lstm_best.ckpt') 
                                best_validation_loss = vl
                                best_step = step
                        #elif loss < best_training_loss: #if no validation set is given, save the model with the losest loss on the training set
                            #best_training_loss = loss
                            #tracking.append([step,loss])                
                            #saver.save(sess, self.path+'lstm_best.ckpt') 
                            #print 'step %d saved' %step
                        acc = sess.run(accuracy, feed_dict={x: train_data.reshape((-1,n_timesteps,n_input)), y: train_labels})
                        print("Iter " + str(step) + ", Epoch Loss= " + \
                              "{:.6f}".format(avg_cost) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
                        if 'x_valid' and 'y_valid' in kwargs:
                            print ("validation loss %.6f" %vl)
                            print ("validation accuracy:", \
                                    sess.run(accuracy, feed_dict={x:x_valid,y:kwargs['y_valid']}))                                               
                        else:
                            tracking.append([step,loss,ow])                                               
                    elif step == training_iters-1:
                        if 'x_test' and 'y_test' in kwargs:
                            test_data = kwargs['test_data'].reshape((-1,n_timesteps,n_input))
                            print("Testing Accuracy:", \
                              sess.run(accuracy, feed_dict={x: test_data, y: kwargs['test_labels'], beta_regul: 0.0}))
                        saver.save(sess, self.path+'lstm_best.ckpt') #if no early stopping save at the end
                        print 'step %d saved' %step
                print("Optimization Finished!")            
                # Calculate accuracy on the test set if one is given
                if (('test_data' in kwargs) and ('test_labels' in kwargs)):
                    test_data = kwargs['test_data'].reshape((-1,n_timesteps,n_input))
                    print("Testing Accuracy:", \
                    sess.run(accuracy, feed_dict={x: test_data, y: kwargs['test_labels']}))
                    pred_proba = sess.run(probabilities, feed_dict={x: test_data, y: kwargs['test_labels']})
                    pred_class = np.argmax(pred_proba,axis=1)
            else:
                saver.restore(sess, self.path+'lstm_best.ckpt')
                train_data = train_data.reshape((-1,n_timesteps,n_input))
                pred_proba = sess.run(probabilities, feed_dict={x: train_data, y: train_labels})
                #since we're only predicting, the inputs are train_data and train_labels
                pred_class = np.argmax(pred_proba,axis=1) 
                if predict_proba==True:
                    return pred_proba
                else:
                    return pred_class
            if is_train:
                return tracking
                    
    def cv_score(self,x_train,y_train,n_input,n_timesteps,n_classes,n_folds,**kwargs):
        #normally you wouldn't use early-stopping here but when you have separate training/validation/test sets
        #get parameter values
        if 'scoring' in kwargs:
            scoring = kwargs['scoring']
        else:
            scoring = metrics.accuracy_score
        if 'random_seed' in self.params:
            kf = model_selection.StratifiedKFold(n_splits=n_folds,shuffle=True,random_state = self.params['random_seed'])
        else:
            kf = model_selection.StratifiedKFold(n_splits=n_folds,shuffle=True)  
        kscores = []
        steps = []
        for train_idx, valid_idx in kf.split(x_train,y_train[:,1]):
            x_train_ = x_train[train_idx]
            y_train_ = y_train[train_idx]
            x_valid = x_train[valid_idx]
            y_valid = y_train[valid_idx]
            tracking = lstm.fit_predict(self,x_train_,y_train_,n_input,n_timesteps,
                            n_classes,is_train=True,predict_proba=False,x_valid=x_valid,y_valid=y_valid)
            valid_prediction = lstm.fit_predict(self,x_valid,y_valid,n_input,n_timesteps,n_classes,is_train=False,predict_proba=False)
            kscores.append(scoring(np.argmax(y_valid,axis=1),valid_prediction))
            tracking = pd.DataFrame(tracking,columns=['step','tloss','vloss','output-weights'])
            bstep = tracking['step'][tracking['vloss']==tracking['vloss'].min()]
            steps.append(bstep.values[0])
        return kscores,steps

        
        

        
        
        
        
        
        