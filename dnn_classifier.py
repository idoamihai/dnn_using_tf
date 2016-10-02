# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:32:35 2016

@author: idoamihai
"""

import tensorflow as tf
import numpy as np, pandas as pd
from sklearn import cross_validation, preprocessing, metrics, grid_search
import random
from tensorflow.python.ops import rnn, rnn_cell


def reformat(dataset, labels, num_labels, example_rows, example_columns):
  dataset = dataset.reshape((-1, example_rows * example_columns)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
  
def grid_search_dnn(param_grid,x_train,y_train,example_rows=1,num_labels=2,n_folds=5,scoring=metrics.accuracy_score):
    max_score = 0
    max_params = {}
    grid = grid_search.ParameterGrid(param_grid)
    grid_search_scores = []
    for params_ in grid:
        clf = dnn(params_)
        score = clf.cv_score(x_train,y_train,example_rows,num_labels,n_folds,scoring=scoring,predict_proba=False)
        if score > max_score:
            max_score = score
            max_params = params_
        dict_ = params_
        dict_['score'] = score
        grid_search_scores.append(dict_) 
    return grid_search_scores,max_params,max_score  
    

def grid_search_lstm(param_grid,x_train,y_train,num_labels=2,n_input=4,n_timesteps=31,n_folds=5,scoring=metrics.accuracy_score):
    max_score = 0
    max_params = {}
    grid = grid_search.ParameterGrid(param_grid)
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
    
class feedforward():
    
    def __init__(self,params,**kwargs):
        self.params = params
        if 'path' in kwargs: #path for saving and restoring models
            self.path = kwargs['path']
        else:
            self.path = ''
    #split training into train and validation sets
    def cv_score(self,x_train,y_train,example_rows,num_labels,n_folds,**kwargs):
        #normally you wouldn't use early-stopping here but when you have separate training/validation/test sets
        #get parameter values
        if 'scoring' in kwargs:
            scoring = kwargs['scoring']
        else:
            scoring = metrics.accuracy_score
        if 'random_seed' in self.params:
            kf = cross_validation.StratifiedKFold(y=y_train,
              n_folds=n_folds,shuffle=True,random_state = self.params['random_seed'])
        else:
            kf = cross_validation.StratifiedKFold(y=y_train,n_folds=n_folds,shuffle=True)  
        kscores = []
        fold = 0
        for train_idx, valid_idx in kf:
            fold += 1
            x_train_ = x_train.iloc[train_idx]
            y_train_ = y_train.iloc[train_idx]
            x_valid = x_train.iloc[valid_idx]
            y_valid = y_train.iloc[valid_idx]
            dnn.fit_predict(self,x_train_,y_train_,example_rows,num_labels,is_train=True)
            valid_prediction = dnn.fit_predict(self,x_valid,y_valid,example_rows,num_labels,is_train=False,predict_proba=False)
            kscores.append(scoring(y_valid,valid_prediction))
        return np.mean(kscores)       
       
    def fit_predict(self,x,y,example_rows,num_labels,is_train,predict_proba=False,**kwargs):
        #refit and predict on test
        if 'patience' in self.params:
            patience = self.params['patience']
            patience_increase = 5 #wait this much longer when a new best is found
            improvement_threshold = 1.0 # a relative improvement of this much is considered significant
        if 'random_seed' in self.params:
            random.seed(self.params['random_seed'])
        if 'evaluation_frequency' in self.params:
            evaluation_frequency = self.params['evaluation_frequency']
        else:
            evaluation_frequency = 10
        example_columns = x.shape[1]
        num_steps = self.params['num_steps']
        hidden = self.params['hidden']
        if 'l2_regul' in self.params:
            l2_regul = self.params['l2_regul']
        else:
            l2_regul = 0.0
        if 'keep_prob' in self.params:
            keep_prob = self.params['keep_prob']
        else:
            keep_prob = 1.0
        if 'keep_prob_input' in self.params:
            keep_prob_input = self.params['keep_prob_input']
        else:
            keep_prob_input = 1.0 
        if 'batch_size' in self.params:
            batch_size = self.params['batch_size']
        train_dataset, train_labels = reformat(np.array(x), np.array(y),num_labels,example_rows,example_columns)
        if 'x_test' and 'y_test' in kwargs:
            x_test, y_test = kwargs['x_test'],kwargs['y_test']
            test_dataset, test_labels = reformat(np.array(x_test), np.array(y_test),num_labels,example_rows,example_columns)
        if 'x_valid' and 'y_valid' in kwargs:
            x_valid, y_valid = kwargs['x_valid'],kwargs['y_valid']
            valid_dataset, valid_labels = reformat(np.array(x_valid), np.array(y_valid),num_labels,example_rows,example_columns)
        if 'scale' in self.params:
            scaler = preprocessing.StandardScaler()
            scaler.fit(train_dataset)        
            train_dataset = scaler.transform(train_dataset)
            if 'x_valid' and 'y_valid' in kwargs:
                valid_dataset = scaler.transform(valid_dataset)
            if 'x_test' and 'y_test' in kwargs:
                test_dataset = scaler.transform(test_dataset)        
        graph = tf.Graph()
        with graph.as_default():
          if 'batch_size' in self.params and is_train==True:
              tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, example_rows * example_columns))
              tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
          else:
              tf_train_dataset = tf.constant(train_dataset)
              tf_train_labels = tf.constant(train_labels)
          if 'x_valid' and 'y_valid' in kwargs:
              tf_valid_dataset = tf.constant(valid_dataset)
              tf_valid_labels = tf.constant(valid_labels)
          if 'x_test' and 'y_test' in kwargs:
              tf_test_dataset = tf.constant(test_dataset)    
          # Variables.
          beta_regul = tf.placeholder(tf.float32)
          weights = {}
          biases = {}
          weights[0] = tf.Variable(tf.truncated_normal([example_rows * example_columns,
                                                      hidden[0]],mean = 0.0,
                                                      stddev= 1.0))
          biases[0] = tf.Variable(tf.zeros([hidden[0]]))
          for i in np.arange(1,len(hidden)):
              weights[i] = tf.Variable(
                tf.truncated_normal([hidden[i-1], hidden[i]],mean = 0.0,
                                    stddev= 1.0))
              biases[i] = tf.Variable(tf.zeros([hidden[i]]))
          weights[len(weights)] = tf.Variable(
                  tf.truncated_normal([hidden[-1], num_labels],mean = 0.0,
                        stddev = 1.0))  
          biases[len(biases)] = tf.Variable(tf.zeros([num_labels]))
          
          # Training computation.
          # We multiply the inputs with the weight matrix, and add biases. We compute
          # the softmax and cross-entropy . We take the average of this
          # cross-entropy across all training examples: that's our loss.
          layer = {}
          dropout = {}
          layer[0] = tf.nn.relu(tf.matmul(tf_train_dataset,weights[0]) + biases[0])
          dropout[0] = tf.nn.dropout(layer[0], keep_prob=keep_prob_input)
          for i in np.arange(1,len(weights)-1):
              if is_train:
                  layer[i] = tf.nn.relu(tf.matmul(dropout[i-1],weights[i]) + biases[i])
                  dropout[i] = tf.nn.dropout(layer[i], keep_prob=keep_prob)
              else:
                  layer[i] = tf.nn.relu(tf.matmul(layer[i-1],weights[i]*keep_prob) + biases[i]) #weight scaling during test
          if is_train:
              logits = tf.matmul(dropout[len(dropout)-1],weights[len(weights)-1]) + biases[len(biases)-1]  
          else:
              logits = tf.matmul(layer[len(layer)-1],weights[len(weights)-1]) + biases[len(biases)-1]
          if num_labels == 2:
              loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits,tf_train_labels))
          else:
              loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) 
          reg = 0
          for i in range(len(weights)):
            reg += beta_regul * tf.nn.l2_loss(weights[i])
          loss = loss + reg
          # Optimizer.
          # We are going to find the minimum of this loss using gradient descent.
          if self.params['learning_rate'] == 'exp_decay':
              global_step = tf.Variable(0, trainable = False)
              learning_rate = tf.train.exponential_decay(
                0.5, global_step, 1000, 0.5, staircase=True) 
              optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                global_step = global_step)       
          else:
              learning_rate = self.params['learning_rate']
              optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)       
          # Predictions for the training, validation, and test data.
          # These are not part of training, but merely here so that we can report
          # accuracy figures as we train.
          if num_labels == 2:
              train_prediction = tf.nn.sigmoid(logits)
          else:
              train_prediction = tf.nn.softmax(logits)
          if 'x_valid' and 'y_valid' in kwargs:
              valid_layer = {}
              valid_layer[0] = tf.nn.relu(tf.matmul(tf_valid_dataset,weights[0])+biases[0])
              for i in np.arange(1,len(weights)-1):
                  valid_layer[i] = tf.nn.relu(tf.matmul(valid_layer[i-1],weights[i]) + biases[i])
              valid_logits = tf.matmul(valid_layer[len(valid_layer)-1],weights[len(weights)-1]) + biases[len(biases)-1]
              if num_labels == 2:
                  valid_prediction = tf.nn.sigmoid(valid_logits)
                  valid_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(valid_logits,tf_valid_labels))
              else:
                  valid_prediction = tf.nn.softmax(valid_logits)
                  valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits,tf_valid_labels))
          #lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset,weights1)+biases1)
          #lay2_valid = tf.nn.relu(tf.matmul(lay1_valid,weights2)+biases2)
          #valid_prediction = tf.nn.softmax(
          #  tf.matmul(lay2_valid, weights3) + biases3)
          if 'x_test' and 'y_test' in kwargs:
              test_layer = {}
              test_layer[0] = tf.nn.relu(tf.matmul(tf_test_dataset,weights[0])+biases[0])
              for i in np.arange(1,len(weights)-1):
                  test_layer[i] = tf.nn.relu(tf.matmul(test_layer[i-1],weights[i]) + biases[i])
              if num_labels == 2:
                  test_prediction = tf.nn.sigmoid(
                    tf.matmul(test_layer[len(test_layer)-1],weights[len(weights)-1]) + biases[len(biases)-1])               
              else:
                  test_prediction = tf.nn.softmax(
                        tf.matmul(test_layer[len(test_layer)-1],weights[len(weights)-1]) + biases[len(biases)-1])                           
        with tf.Session(graph=graph) as session:
          # This is a one-time operation which ensures the parameters get initialized as
          # we described in the graph: random weights for the matrix, zeros for the
          # biases. 
          tf.initialize_all_variables().run() 
          saver = tf.train.Saver()
          print('Initialized')
          best_validation_loss = np.inf
          best_training_loss = np.inf
          best_step = 0
          patience_steps = 0
          tracking = []
          for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            if is_train:
                if 'batch_size' in self.params:
                    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                    batch_data = train_dataset[offset:(offset + batch_size), :]
                    batch_labels = train_labels[offset:(offset + batch_size), :]
                    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regul: l2_regul}
                else:
                    feed_dict = {beta_regul: l2_regul}
                if (step % evaluation_frequency == 0):
                    if 'x_valid' and 'y_valid' in kwargs:
                        _, l, predictions, vl, vp = session.run([optimizer, loss, train_prediction, 
                                 valid_loss, valid_prediction],
                            feed_dict = feed_dict)
                        tracking.append([step,l,vl]) #track the model step loss
                    else:
                        _, l, predictions = session.run([optimizer, loss, train_prediction],
                            feed_dict = feed_dict)
                        tracking.append([step,l])
                    if 'x_valid' and 'y_valid' in kwargs:
                        if 'patience' in self.params:
                          if vl < best_validation_loss:
                              saver.save(session, self.path+'model_best.ckpt')
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
                                      saver.restore(session, self.path+'model_best.ckpt')
                                      print('Test accuracy: %.1f%%' % accuracy(
                                               test_prediction.eval(),test_labels))
                                  break  
                        elif vl < best_validation_loss:
                              best_validation_loss = vl
                              best_step = step
                              saver.save(session, self.path+'model_best.ckpt')  
                    #elif l < best_training_loss:
                    #      best_training_loss = l
                    #      saver.save(session, self.path+'model_best.ckpt')                                                                 
                    print('Loss at step %d: %f' % (step, l)) 
                    if 'batch_size' in self.params:
                      print('Training accuracy: %.1f%%' % accuracy(
                        predictions, batch_labels))
                    else:
                      print('Training accuracy: %.1f%%' % accuracy(
                        predictions, train_labels))
                    if 'x_valid' and 'y_valid' in kwargs:
                      print('Validation Loss at step %d: %f' % (step, vl))
                      print('Validation accuracy: %.1f%%' % accuracy(
                                 vp, valid_labels))
                elif step == num_steps-1:
                  if 'x_test' and 'y_test' in kwargs:
                      print('Test accuracy: %.1f%%' % accuracy(
                               test_prediction.eval(),test_labels)) 
                  saver.save(session, self.path+'model_best.ckpt') #if no early stopping save at the end    
            else:
                saver.restore(session, self.path+'model_best.ckpt')
                pred_proba = train_prediction.eval()
                pred_class = np.argmax(pred_proba,axis=1) 
                if predict_proba==True:
                    return pred_proba
                else:
                    return pred_class
        if is_train:
            return tracking
  
class lstm():        
    def __init__(self,params,**kwargs):
        self.params = params
        if 'path' in kwargs: #path for saving and restoring models
            self.path = kwargs['path']
        else:
            self.path = ''

    def fit_predict(self,train_data,train_labels,n_input,n_timesteps,n_classes,is_train,predict_proba=False,clipping=False,**kwargs):
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
               0.001, global_step, 10000, 0.96, staircase=True) 
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
            if 'l2_regul' in self.params:
                beta_regul = self.params['l2_regul']
            else:
                beta_regul = 0
            n_classes = n_classes             
            # tf Graph input
            x = tf.placeholder("float", [None, n_timesteps, n_input])
            y = tf.placeholder("float", [None, n_classes])            
            # Define weights
            weights = {
                'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
            }
            biases = {
                'out': tf.Variable(tf.random_normal([n_classes]))
            }   

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
            
            pred, output_weights = BiRNN(x, weights, biases)

            '''        
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
                return pred           
            pred = RNN(x, weights, biases) 
            '''
            if n_classes == 2:
                probabilities = tf.nn.sigmoid(pred) #not exactly probabilities but often close enough
                cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y)) +\
                beta_regul * tf.nn.l2_loss(weights['out'])
            else:
                probabilities = tf.nn.softmax(pred)
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) +\
                beta_regul * tf.nn.l2_loss(weights['out'])
            # Define loss and optimizer
            opt_function = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gradients = opt_function.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, clip_value_min = -5.0,clip_value_max=5.0), var) for grad, var in gradients]
            if clipping:
                optimizer = opt_function.apply_gradients(capped_gradients)
            else:
                optimizer = opt_function.apply_gradients(gradients)
            # Evaluate model
            correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))            
            # Initializing the variables
            init = tf.initialize_all_variables()        
        # Launch the graph
        with tf.Session(graph=graph) as sess:
            sess.run(init)
            saver = tf.train.Saver()
            step = 1
            # Keep training until reach max iterations
            if is_train:
                best_validation_loss = np.inf
                best_training_loss = np.inf
                best_step = 0
                patience_steps = 0
                tracking = []
                for step in range(training_iters):
                    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                    batch_x = train_data[offset:(offset + batch_size), :]
                    batch_y = train_labels[offset:(offset + batch_size), :]
                    batch_x = batch_x.reshape((batch_size, n_timesteps, n_input))
                    # Run optimization op (backprop)
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    ow = sess.run(output_weights, feed_dict={x: batch_x, y: batch_y})
                    if step % display_step == 0:                        
                        if 'x_valid' and 'y_valid' in kwargs:
                            x_valid = kwargs['x_valid'].reshape((-1,n_timesteps,n_input))
                            vl = sess.run(cost,feed_dict={x:x_valid,y:kwargs['y_valid']})
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
                        else:
                            tracking.append([step,loss,ow])                                               
                        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                        print("Iter " + str(step) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
                        if 'x_valid' and 'y_valid' in kwargs:
                            print ("validation loss %.6f" %vl)
                            print ("validation accuracy:", \
                                    sess.run(accuracy, feed_dict={x:x_valid,y:kwargs['y_valid']}))
                    elif step == training_iters-1:
                        if 'x_test' and 'y_test' in kwargs:
                            test_data = kwargs['test_data'].reshape((-1,n_timesteps,n_input))
                            print("Testing Accuracy:", \
                              sess.run(accuracy, feed_dict={x: test_data, y: kwargs['test_labels']}))
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
            kf = cross_validation.StratifiedKFold(y=np.argmax(y_train,axis=1),
              n_folds=n_folds,shuffle=True,random_state = self.params['random_seed'])
        else:
            kf = cross_validation.StratifiedKFold(y=np.argmax(y_train,axis=1),n_folds=n_folds,shuffle=True)  
        kscores = []
        steps = []
        fold = 0
        for train_idx, valid_idx in kf:
            fold += 1
            x_train_ = x_train[train_idx]
            y_train_ = y_train[train_idx]
            x_valid = x_train[valid_idx]
            y_valid = y_train[valid_idx]
            tracking = lstm.fit_predict(self,x_train_,y_train_,n_input,n_timesteps,
                            n_classes,is_train=True,predict_proba=False,x_valid=x_valid,y_valid=y_valid)
            valid_prediction = lstm.fit_predict(self,x_valid,y_valid,n_input,n_timesteps,n_classes,is_train=False,predict_proba=False)
            kscores.append(scoring(np.argmax(y_valid,axis=1),valid_prediction))
            tracking = pd.DataFrame(tracking,columns=['step','tloss','vloss'])
            bstep = tracking['step'][tracking['vloss']==tracking['vloss'].min()]
            steps.append(bstep.values[0])
        return np.mean(kscores),np.mean(steps) 

        
        

        
        
        
        
        
        