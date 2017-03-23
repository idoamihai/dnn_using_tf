from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
import matplotlib.pyplot as plt

def import_mnist(limit=10000):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    x_train = mnist.train.images[:limit,:]
    y_train = mnist.train.labels[:limit,:]
    x_valid = mnist.validation.images[:limit,:]
    y_valid = mnist.validation.labels[:limit,:]
    x_test = mnist.test.images[:limit,:]
    y_test = mnist.test.labels[:limit,:]
    # Network Parameters
    return x_train,y_train,x_valid,y_valid,x_test,y_test

def vectorize_labels(y):
    n_classes = len(set(y))
    y_vect = np.zeros([len(y),n_classes])
    y_vect[np.arange(len(y)), y.astype(int)] = 1
    return y_vect


def batch_data(x,batch_size):             
    idx = range(len(x))
    np.random.shuffle(idx)
    x = x[idx]     
    total_batches = int(len(x)/batch_size)
    batch_x = np.array_split(x,total_batches)
    batched_data = iter(batch_x) 
    return batched_data

def plot_reconstructions(original_images,decoded_images,examples_to_show,image_dims):
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, examples_to_show, 
                        figsize=(examples_to_show, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(original_images[i], (image_dims[0], image_dims[1])))
        a[1][i].imshow(np.reshape(decoded_images[i], (image_dims[0], image_dims[1])))
    f.show()
    plt.draw()
            

class autoencoder_sparse:
    
    def __init__(self):
        self.learning_rate = 0.01
        self.training_iters = 20
        self.batch_size = 256
        self.display_step = 1
        self.examples_to_show = 10        
        self.n_hidden_1 = 256 # 1st layer num features
        self.n_hidden_2 = 128 # 2nd layer num features
        self.path = os.getcwd()
        
    def dump(self):
        return dict(self.__dict__)
    
    def run_mnist(self):
        x_train,y_train,x_valid,y_valid,x_test,y_test = import_mnist()
        mnist_preds = autoencoder_sparse.build_model(self,x_train,
                                            run_test=True,
                                            x_test=x_test)
        return mnist_preds
    
    def build_model(self,x_train,**kwargs):
            n_input = x_train.shape[1]
            params = autoencoder_sparse.dump(self)
            params.update(kwargs)  
            graph = tf.Graph()
            n_hidden_1,n_hidden_2 = params['n_hidden_1'],params['n_hidden_2']
            with graph.as_default():
                X = tf.placeholder("float", [None, n_input])
                weights = {
                    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
                    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
                    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
                }
                biases = {
                    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
                    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
                    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
                    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
                }                              
                # Building the encoder
                def encoder(x):
                    # Encoder Hidden layer with sigmoid activation #1
                    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                                   biases['encoder_b1']))
                    # Decoder Hidden layer with sigmoid activation #2
                    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                                   biases['encoder_b2']))
                    return layer_2                               
                # Building the decoder
                def decoder(x):
                    # Encoder Hidden layer with sigmoid activation #1
                    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                                   biases['decoder_b1']))
                    # Decoder Hidden layer with sigmoid activation #2
                    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                                   biases['decoder_b2']))
                    return layer_2
                
                # Construct model
                encoder_op = encoder(X)
                decoder_op = decoder(encoder_op)
                
                # Prediction
                y_pred = decoder_op
                # Targets (Labels) are the input data.
                y_true = X
                
                # Define loss and optimizer, minimize the squared error
                cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
                optimizer = tf.train.RMSPropOptimizer(params['learning_rate']).minimize(cost)
                
                # Initializing the variables
                init = tf.global_variables_initializer()
                
                # Launch the graph
                with tf.Session() as sess:
                    if 'x_test' and 'y_test' in params:
                        x_test = params['x_test']
                    else:
                         x_test = x_train                 
                    saver = tf.train.Saver()
                    summary_writer = tf.summary.FileWriter(logdir=params['path']+'/logdir/autoe_train', graph=tf.get_default_graph())
                    sess.run(init)
                    # Training cycle
                    for iteration in range(params['training_iters']):
                        total_batches = int(len(x_train)/params['batch_size'])
                        batched_data = batch_data(x_train,params['batch_size'])
                        for batch in range(total_batches):
                            batch_x = batched_data.next()
                            # Run optimization op (backprop) and cost op (to get loss value)
                            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x})
                        # Display logs per epoch step
                        if iteration % params['display_step'] == 0:
                            print("Epoch:", '%04d' % (iteration+1),
                                  "cost=", "{:.9f}".format(c))                
                    print("Optimization Finished!")                
                    # Applying encode and decode over test set
                    encode_decode = sess.run(
                        y_pred, feed_dict={X: x_test})
                    return encode_decode


            


