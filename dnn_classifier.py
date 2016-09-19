# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:32:35 2016

@author: idoamihai
"""

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pickle, os, random
from sklearn import cross_validation, preprocessing, metrics, grid_search
import tensorflow.contrib.learn as skflow


os.chdir('/Users/idoamihai/Documents/Bandmanage/falling/fall_detection')

x_train,y_train,x_test,y_test = pickle.load(open( "data_lstm.p", "rb" ))

params = {'num_steps':1001,'hidden':[10,10],'l2_regul':1e-2,'keep_prob':0.5,
        'n_folds': 5}
    
       
features = x_train.columns
       
x_train = x_train[features] 
x_test = x_test[features]

example_rows = 1
example_columns = x_train.shape[1]
num_labels=2
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, example_rows * example_columns)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

class dnn():
    
    def __init__(self,params):
        self.params = params
    #split training into train and validation sets
    def cv_score(self,x_train,y_train,example_rows,num_labels,n_folds,**kwargs):
        #normally you wouldn't use early-stopping here but when you have separate training/validation/test sets
        #get parameter values
        if 'patience' in self.params:
            patience = self.params['patience']
            patience_increase = 2 #wait this much longer when a new best is found
            improvement_threshold = 0.995 # a relative improvement of this much is considered significant
        if 'random_seed' in self.params:
            random.seed(self.params['random_seed'])
        example_columns = x_train.shape[1]
        num_steps = self.params['num_steps']
        hidden = self.params['hidden'] #number of units per layer as list
        l2_regul = self.params['l2_regul'] #give 0 for no l2 regularization
        keep_prob = self.params['keep_prob'] #give 1 for no dropout
        if 'random_seed' in self.params:
            kf = cross_validation.StratifiedKFold(y=y_train,
                              n_folds=n_folds,shuffle=True,random_state = params['random_seed'])
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
            
            train_dataset, train_labels = reformat(np.array(x_train_), np.array(y_train_))
            valid_dataset, valid_labels = reformat(np.array(x_valid), np.array(y_valid))
            #test_dataset, test_labels = reformat(np.array(x_test), np.array(y_test))
            
            if 'scale' in self.params:
                scaler = preprocessing.StandardScaler()
                scaler.fit(train_dataset)        
                train_dataset = scaler.transform(train_dataset)
                valid_dataset = scaler.transform(valid_dataset)
                #test_dataset = scaler.transform(test_dataset)
        
            graph = tf.Graph()
            with graph.as_default():
              tf_train_dataset = tf.constant(train_dataset)
              tf_train_labels = tf.constant(train_labels)
              tf_valid_dataset = tf.constant(valid_dataset)
              tf_valid_labels = tf.constant(valid_labels)
              beta_regul = tf.placeholder(tf.float32)
              
              # Variables.
              weights = {}
              biases = {}
              weights[0] = tf.Variable(tf.truncated_normal([example_rows * example_columns,
                                                          hidden[0]],
                                                          stddev= 0.1))
              biases[0] = tf.Variable(tf.zeros([hidden[0]]))
              for i in np.arange(1,len(hidden)):
                  weights[i] = tf.Variable(
                    tf.truncated_normal([hidden[i-1], hidden[i]],
                                        stddev= 0.1))
                  biases[i] = tf.Variable(tf.zeros([hidden[i]]))
              weights[len(weights)] = tf.Variable(
                      tf.truncated_normal([hidden[-1], num_labels],
                            stddev= 0.1))  
              biases[len(biases)] = tf.Variable(tf.zeros([num_labels]))
              # Training computation.
              # We multiply the inputs with the weight matrix, and add biases. We compute
              # the softmax and cross-entropy . We take the average of this
              # cross-entropy across all training examples: that's our loss.
              layer = {}
              dropout = {}
              layer[0] = tf.nn.relu(tf.matmul(tf_train_dataset,weights[0]) + biases[0])
              dropout[0] = tf.nn.dropout(layer[0], keep_prob=keep_prob, seed=0)
              for i in np.arange(1,len(weights)-1):
                  layer[i] = tf.nn.relu(tf.matmul(dropout[i-1],weights[i]) + biases[i])
                  dropout[i] = tf.nn.dropout(layer[i], keep_prob=keep_prob, seed=0)
              logits = tf.matmul(dropout[len(dropout)-1],weights[len(weights)-1]) + biases[len(biases)-1]     
                  
              loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) 
              reg = 0
              for i in range(len(weights)):
                reg += beta_regul * tf.nn.l2_loss(weights[i])
              loss = loss + reg
              # Optimizer.
              # We are going to find the minimum of this loss using gradient descent.
              global_step = tf.Variable(0)
              learning_rate = tf.train.exponential_decay(
                0.5, global_step, 1000, 0.5, staircase=True)
            
              optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                global_step = global_step)
              
              # Predictions for the training, validation, and test data.
              # These are not part of training, but merely here so that we can report
              # accuracy figures as we train.
              train_prediction = tf.nn.softmax(logits)
              valid_layer = {}
              valid_layer[0] = tf.nn.relu(tf.matmul(tf_valid_dataset,weights[0])+biases[0])
              for i in np.arange(1,len(weights)-1):
                  valid_layer[i] = tf.nn.relu(tf.matmul(valid_layer[i-1],weights[i]) + biases[i])
              valid_logits = tf.matmul(valid_layer[len(valid_layer)-1],weights[len(weights)-1]) + biases[len(biases)-1]
              valid_prediction = tf.nn.softmax(valid_logits)
              valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits,tf_valid_labels))
              
            
            with tf.Session(graph=graph) as session:
              # This is a one-time operation which ensures the parameters get initialized as
              # we described in the graph: random weights for the matrix, zeros for the
              # biases. 
              tf.initialize_all_variables().run()
              print('Initialized')
              best_validation_loss = np.inf
              patience_steps = 0
              for step in range(num_steps):
                # Run the computations. We tell .run() that we want to run the optimizer,
                # and get the loss value and the training predictions returned as numpy
                # arrays.
                _, l, predictions,vl = session.run([optimizer, loss, train_prediction, valid_loss],
                    feed_dict = {beta_regul: l2_regul})
                if (step % 10 == 0):
                  print('Loss at step %d: %f' % (step, l))
                  print('Training accuracy: %.1f%%' % accuracy(
                    predictions, train_labels))
                  # Calling .eval() on valid_prediction is basically like calling run(), but
                  # just to get that one numpy array. Note that it recomputes all its graph
                  # dependencies.
                  print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))
                  print('Validation Loss at step %d: %f' % (step, vl))
                  if 'patience' in self.params:
                      if vl - best_validation_loss >= improvement_threshold:
                          best_validation_loss = vl
                          patience_steps = 0
                      else:
                          patience_steps += 1
                          if ((patience_steps == patience_increase) and (step >= patience)):
                              kf_pred_proba = valid_prediction.eval()
                              kf_pred = np.argmax(kf_pred_proba,axis=1)
                              kscores.append(metrics.accuracy_score(y_valid,kf_pred))
                              break   
                  #proba = valid_prediction.eval()
                  #p = np.argmax(proba,axis=1)
                  #f1 = metrics.f1_score(valid_labels[:,1],p)
                  #print ('f1 score: %.1f%%' % f1)
                kf_pred_proba = valid_prediction.eval()
                kf_pred = np.argmax(kf_pred_proba,axis=1)
              #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
              #test_pred_proba = test_prediction.eval()
              #test_pred = np.argmax(test_pred_proba,axis=1)
              kscores.append(metrics.accuracy_score(y_valid,kf_pred))
        return np.mean(kscores)
        
    def test_score(self,x_train,y_train,x_test,y_test,example_rows,num_labels,**kwargs):
        #refit and predict on test
        if 'patience' in self.params:
            patience = self.params['patience']
            patience_increase = 2 #wait this much longer when a new best is found
            improvement_threshold = 0.995 # a relative improvement of this much is considered significant
        if 'random_seed' in self.params:
            random.seed(self.params['random_seed'])
        example_columns = x_train.shape[1]
        num_steps = self.params['num_steps']
        hidden = self.params['hidden']
        l2_regul = self.params['l2_regul']
        keep_prob = self.params['keep_prob']
        train_dataset, train_labels = reformat(np.array(x_train), np.array(y_train))
        if 'x_valid' and 'y_valid' in kwargs:
            x_valid, y_valid = kwargs['x_valid'],kwargs['y_valid']
            valid_dataset, valid_labels = reformat(np.array(x_valid), np.array(y_valid))
        test_dataset, test_labels = reformat(np.array(x_test), np.array(y_test))
        if 'scale' in self.params:
            scaler = preprocessing.StandardScaler()
            scaler.fit(train_dataset)        
            train_dataset = scaler.transform(train_dataset)
            if 'x_valid' and 'y_valid' in kwargs:
                valid_dataset = scaler.transform(valid_dataset)
            test_dataset = scaler.transform(test_dataset)
        
        graph = tf.Graph()
        with graph.as_default():
          tf_train_dataset = tf.constant(train_dataset)
          tf_train_labels = tf.constant(train_labels)
          if 'x_valid' and 'y_valid' in kwargs:
              tf_valid_dataset = tf.constant(valid_dataset)
              tf_valid_labels = tf.constant(valid_labels)
          tf_test_dataset = tf.constant(test_dataset)
          beta_regul = tf.placeholder(tf.float32)
    
          # Variables.
          weights = {}
          biases = {}
          weights[0] = tf.Variable(tf.truncated_normal([example_rows * example_columns,
                                                      hidden[0]],
                                                      stddev= 0.1))
          biases[0] = tf.Variable(tf.zeros([hidden[0]]))
          for i in np.arange(1,len(hidden)):
              weights[i] = tf.Variable(
                tf.truncated_normal([hidden[i-1], hidden[i]],
                                    stddev= 0.1))
              biases[i] = tf.Variable(tf.zeros([hidden[i]]))
          weights[len(weights)] = tf.Variable(
                  tf.truncated_normal([hidden[-1], num_labels],
                        stddev= 0.1))  
          biases[len(biases)] = tf.Variable(tf.zeros([num_labels]))
          
          # Training computation.
          # We multiply the inputs with the weight matrix, and add biases. We compute
          # the softmax and cross-entropy . We take the average of this
          # cross-entropy across all training examples: that's our loss.
          layer = {}
          dropout = {}
          layer[0] = tf.nn.relu(tf.matmul(tf_train_dataset,weights[0]) + biases[0])
          dropout[0] = tf.nn.dropout(layer[0], keep_prob=keep_prob, seed=0)
          for i in np.arange(1,len(weights)-1):
              layer[i] = tf.nn.relu(tf.matmul(dropout[i-1],weights[i]) + biases[i])
              dropout[i] = tf.nn.dropout(layer[i], keep_prob=keep_prob, seed=0)
          logits = tf.matmul(dropout[len(dropout)-1],weights[len(weights)-1]) + biases[len(biases)-1]     
              
          loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) 
          reg = 0
          for i in range(len(weights)):
            reg += beta_regul * tf.nn.l2_loss(weights[i])
          loss = loss + reg
          
          # Optimizer.
          # We are going to find the minimum of this loss using gradient descent.
          global_step = tf.Variable(0)
          learning_rate = tf.train.exponential_decay(
            0.5, global_step, 1000, 0.5, staircase=True)
        
          optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
            global_step = global_step)
          
          # Predictions for the training, validation, and test data.
          # These are not part of training, but merely here so that we can report
          # accuracy figures as we train.
          train_prediction = tf.nn.softmax(logits)
          if 'x_valid' and 'y_valid' in kwargs:
              valid_layer = {}
              valid_layer[0] = tf.nn.relu(tf.matmul(tf_valid_dataset,weights[0])+biases[0])
              for i in np.arange(1,len(weights)-1):
                  valid_layer[i] = tf.nn.relu(tf.matmul(valid_layer[i-1],weights[i]) + biases[i])
              valid_logits = tf.matmul(valid_layer[len(valid_layer)-1],weights[len(weights)-1]) + biases[len(biases)-1]
              valid_prediction = tf.nn.softmax(valid_logits)
              valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits,tf_valid_labels))
          #lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset,weights1)+biases1)
          #lay2_valid = tf.nn.relu(tf.matmul(lay1_valid,weights2)+biases2)
          #valid_prediction = tf.nn.softmax(
          #  tf.matmul(lay2_valid, weights3) + biases3)
          test_layer = {}
          test_layer[0] = tf.nn.relu(tf.matmul(tf_test_dataset,weights[0])+biases[0])
          for i in np.arange(1,len(weights)-1):
              test_layer[i] = tf.nn.relu(tf.matmul(test_layer[i-1],weights[i]) + biases[i])
          test_prediction = tf.nn.softmax(
                tf.matmul(test_layer[len(test_layer)-1],weights[len(weights)-1]) + biases[len(biases)-1])         
        
        with tf.Session(graph=graph) as session:
          # This is a one-time operation which ensures the parameters get initialized as
          # we described in the graph: random weights for the matrix, zeros for the
          # biases. 
          tf.initialize_all_variables().run()
          print('Initialized')
          best_validation_loss = np.inf
          patience_steps = 0
          for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            if 'x_valid' and 'y_valid' in kwargs:
                _, l, predictions, vl = session.run([optimizer, loss, train_prediction, valid_loss],
                    feed_dict = {beta_regul: l2_regul})
            else:
                _, l, predictions = session.run([optimizer, loss, train_prediction],
                    feed_dict = {beta_regul: l2_regul})
            if (step % 10 == 0):
              print('Loss at step %d: %f' % (step, l))
              print('Training accuracy: %.1f%%' % accuracy(
                predictions, train_labels))
              #print('Validation accuracy: %.1f%%' % accuracy(
              #      valid_prediction.eval(), valid_labels))
              if 'patience' in self.params:
                  if vl - best_validation_loss >= improvement_threshold:
                      best_validation_loss = vl
                      patience_steps = 0
                  else:
                      patience_steps += 1
                      if ((patience_steps == patience_increase) and (step >= patience)):
                          #val_pred_proba = valid_prediction.eval()
                          #val_pred = np.argmax(kf_pred_proba,axis=1)
                          #test_pred_proba = test_predictions.eval()
                          #test_pred = np.argmax(test_pred_proba,axis=1)
                          score = accuracy(test_prediction.eval(), test_labels)
                          print('Test accuracy: %.1f%%' % score)
                          break  
          score = accuracy(test_prediction.eval(), test_labels)
          print('Test accuracy: %.1f%%' % score)
          #val_pred_proba = valid_prediction.eval()
          #val_pred = np.argmax(kf_pred_proba,axis=1)
          #test_pred_proba = test_prediction.eval()
          #test_pred = np.argmax(test_pred_proba,axis=1)
        return score
        
    def fit_predict(self,x_train,y_train,example_rows,num_labels,is_train,**kwargs):
        #refit and predict on test
        if 'patience' in self.params:
            patience = self.params['patience']
            patience_increase = 2 #wait this much longer when a new best is found
            improvement_threshold = 0.995 # a relative improvement of this much is considered significant
        if 'random_seed' in self.params:
            random.seed(self.params['random_seed'])
        example_columns = x_train.shape[1]
        num_steps = self.params['num_steps']
        hidden = self.params['hidden']
        l2_regul = self.params['l2_regul']
        keep_prob = self.params['keep_prob']
        train_dataset, train_labels = reformat(np.array(x_train), np.array(y_train))
        if 'x_valid' and 'y_valid' in kwargs:
            x_valid, y_valid = kwargs['x_valid'],kwargs['y_valid']
            valid_dataset, valid_labels = reformat(np.array(x_valid), np.array(y_valid))
        test_dataset, test_labels = reformat(np.array(x_test), np.array(y_test))
        if 'scale' in self.params:
            scaler = preprocessing.StandardScaler()
            scaler.fit(train_dataset)        
            train_dataset = scaler.transform(train_dataset)
            if 'x_valid' and 'y_valid' in kwargs:
                valid_dataset = scaler.transform(valid_dataset)
            test_dataset = scaler.transform(test_dataset)
        
        graph = tf.Graph()
        with graph.as_default():
          tf_train_dataset = tf.constant(train_dataset)
          tf_train_labels = tf.constant(train_labels)
          if 'x_valid' and 'y_valid' in kwargs:
              tf_valid_dataset = tf.constant(valid_dataset)
              tf_valid_labels = tf.constant(valid_labels)
          tf_test_dataset = tf.constant(test_dataset)
          beta_regul = tf.placeholder(tf.float32)
    
          # Variables.
          weights = {}
          biases = {}
          weights[0] = tf.Variable(tf.truncated_normal([example_rows * example_columns,
                                                      hidden[0]],
                                                      stddev= 0.1))
          biases[0] = tf.Variable(tf.zeros([hidden[0]]))
          for i in np.arange(1,len(hidden)):
              weights[i] = tf.Variable(
                tf.truncated_normal([hidden[i-1], hidden[i]],
                                    stddev= 0.1))
              biases[i] = tf.Variable(tf.zeros([hidden[i]]))
          weights[len(weights)] = tf.Variable(
                  tf.truncated_normal([hidden[-1], num_labels],
                        stddev= 0.1))  
          biases[len(biases)] = tf.Variable(tf.zeros([num_labels]))
          
          # Training computation.
          # We multiply the inputs with the weight matrix, and add biases. We compute
          # the softmax and cross-entropy . We take the average of this
          # cross-entropy across all training examples: that's our loss.
          layer = {}
          dropout = {}
          layer[0] = tf.nn.relu(tf.matmul(tf_train_dataset,weights[0]) + biases[0])
          dropout[0] = tf.nn.dropout(layer[0], keep_prob=keep_prob, seed=0)
          for i in np.arange(1,len(weights)-1):
              layer[i] = tf.nn.relu(tf.matmul(dropout[i-1],weights[i]) + biases[i])
              dropout[i] = tf.nn.dropout(layer[i], keep_prob=keep_prob, seed=0)
          logits = tf.matmul(dropout[len(dropout)-1],weights[len(weights)-1]) + biases[len(biases)-1]     
              
          loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) 
          reg = 0
          for i in range(len(weights)):
            reg += beta_regul * tf.nn.l2_loss(weights[i])
          loss = loss + reg
          
          # Optimizer.
          # We are going to find the minimum of this loss using gradient descent.
          global_step = tf.Variable(0)
          learning_rate = tf.train.exponential_decay(
            0.5, global_step, 1000, 0.5, staircase=True)
        
          optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
            global_step = global_step)
          
          # Predictions for the training, validation, and test data.
          # These are not part of training, but merely here so that we can report
          # accuracy figures as we train.
          train_prediction = tf.nn.softmax(logits)
          if 'x_valid' and 'y_valid' in kwargs:
              valid_layer = {}
              valid_layer[0] = tf.nn.relu(tf.matmul(tf_valid_dataset,weights[0])+biases[0])
              for i in np.arange(1,len(weights)-1):
                  valid_layer[i] = tf.nn.relu(tf.matmul(valid_layer[i-1],weights[i]) + biases[i])
              valid_logits = tf.matmul(valid_layer[len(valid_layer)-1],weights[len(weights)-1]) + biases[len(biases)-1]
              valid_prediction = tf.nn.softmax(valid_logits)
              valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits,tf_valid_labels))
          #lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset,weights1)+biases1)
          #lay2_valid = tf.nn.relu(tf.matmul(lay1_valid,weights2)+biases2)
          #valid_prediction = tf.nn.softmax(
          #  tf.matmul(lay2_valid, weights3) + biases3)
          test_layer = {}
          test_layer[0] = tf.nn.relu(tf.matmul(tf_test_dataset,weights[0])+biases[0])
          for i in np.arange(1,len(weights)-1):
              test_layer[i] = tf.nn.relu(tf.matmul(test_layer[i-1],weights[i]) + biases[i])
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
          patience_steps = 0
          for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            if is_train:
                if 'x_valid' and 'y_valid' in kwargs:
                    _, l, predictions, vl = session.run([optimizer, loss, train_prediction, valid_loss],
                        feed_dict = {beta_regul: l2_regul})
                else:
                    _, l, predictions = session.run([optimizer, loss, train_prediction],
                        feed_dict = {beta_regul: l2_regul})
                if (step % 10 == 0):
                  print('Loss at step %d: %f' % (step, l))
                  
                  print('Training accuracy: %.1f%%' % accuracy(
                    predictions, train_labels))
                  #print('Validation accuracy: %.1f%%' % accuracy(
                  if 'patience' in self.params:
                      if vl - best_validation_loss >= improvement_threshold:
                          best_validation_loss = vl
                          patience_steps = 0
                      else:
                          patience_steps += 1
                          if ((patience_steps == patience_increase) and (step >= patience)):
                              saver.save(session, 'model_test.ckpt')
                              break  
                  else:
                      saver.save(session, 'model_test.ckpt')
            else:
                saver.restore(session, 'model_test.ckpt')
                test_pred_proba = test_prediction.eval()
                test_pred = np.argmax(test_pred_proba,axis=1) 
                return test_pred
  
        
def grid_search_dnn(param_grid):
    max_score = 0
    max_params = {}
    grid = grid_search.ParameterGrid(param_grid)
    grid_search_scores = []
    for params_ in grid:
        clf = dnn(params_)
        score = clf.cv_score(x_train,y_train,example_rows=1,num_labels=2,n_folds=5)
        if score > max_score:
            max_score = score
            max_params = params_
        dict_ = params_
        dict_['score'] = score
        grid_search_scores.append(dict_) 
    return grid_search_scores,max_params,max_score

        
        
'''       
param_grid = {'keep_prob': [0.4,0.5, 0.6], 
'hidden': [[10],[50],[50,10]],
'l2_regul':[1e-2,1e-3],'num_steps':[51],'scale':[True]}
'''                                   

        
        
        
        
        
        