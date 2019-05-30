#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:48:05 2019

@author: shalabhmittal
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



aapl = pd.read_excel('AAPL.xlsx')
aapl['Close '] = aapl['Close '].iloc[2:-1]
aapl.index = aapl['Date']

data = aapl['Close '].dropna()
data_to_use = data.values
#data_to_use = data_to_use.set_index(aapl['Date'])
#ytest2 = data_to_use[563:]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_use.reshape(-1, 1))


plt.figure(figsize=(12,7), frameon=False, facecolor='brown', edgecolor='blue')
plt.title('Scaled Apple stocks from August 2014 to August 2017')
plt.xlabel('Days')
plt.ylabel('Scaled value of stocks')
plt.plot(scaled_data, label='Stocks data')
plt.legend()
plt.show()


def lstm_tf(scaledData):
    
    def window_data(data, window_size):
        '''
        This function is used to create Features and Labels datasets. By windowing the data.
        
        Input: data - dataset used in the project
               window_size - how many data points we are going to use to predict the next datapoint in the sequence 
                           [Example: if window_size = 1 we are going to use only the previous day to predict todays stock prices]
        
        Outputs: X - features splitted into windows of datapoints (if window_size = 1, X = [len(data)-1, 1])
                 y - 'labels', actually this is the next number in the sequence, this number we are trying to predict
        '''
        X = []
        y = []
        
        i = 0
        while (i + window_size) <= len(data) - 1:
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
            
            i += 1
        assert len(X) ==  len(y)
        return X, y
    
    X, y = window_data(scaled_data, 7)
    
    X_train  = np.array(X[:556])
    y_train = np.array(y[:556])
    
    X_test = np.array(X[556:])
    y_test = np.array(y[556:])
    print(" ")
    print("LSTM")
    print("X_train size: {}".format(X_train.shape))
    print("y_train size: {}".format(y_train.shape))
    print("X_test size: {}".format(X_test.shape))
    print("y_test size: {}".format(y_test.shape))
    
    #Hyperparameters used in the network
    batch_size = 7 #how many windows of data we are passing at once
    window_size = 7 #how big window_size is (Or How many days do we consider to predict next point in the sequence)
    hidden_layer = 256 #How many units do we use in LSTM cell
    clip_margin = 4 #To prevent exploding gradient, we use clipper to clip gradients below -margin or above this margin
    learning_rate = 0.001 
    epochs = 50
    
    
    inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1])
    targets = tf.placeholder(tf.float32, [batch_size, 1])
    
    # LSTM weights
    #Weights for the input gate
    weights_input_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
    weights_input_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
    bias_input = tf.Variable(tf.zeros([hidden_layer]))
    
    #weights for the forgot gate
    weights_forget_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
    weights_forget_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
    bias_forget = tf.Variable(tf.zeros([hidden_layer]))
    
    #weights for the output gate
    weights_output_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
    weights_output_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
    bias_output = tf.Variable(tf.zeros([hidden_layer]))
    
    #weights for the memory cell
    weights_memory_cell = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
    weights_memory_cell_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
    bias_memory_cell = tf.Variable(tf.zeros([hidden_layer]))
    
    ## Output layer weigts
    weights_output = tf.Variable(tf.truncated_normal([hidden_layer, 1], stddev=0.05))
    bias_output_layer = tf.Variable(tf.zeros([1]))
    
    
    def LSTM_cell(input, output, state):
        
        input_gate = tf.sigmoid(tf.matmul(input, weights_input_gate) + tf.matmul(output, weights_input_hidden) + bias_input)
        
        forget_gate = tf.sigmoid(tf.matmul(input, weights_forget_gate) + tf.matmul(output, weights_forget_hidden) + bias_forget)
        
        output_gate = tf.sigmoid(tf.matmul(input, weights_output_gate) + tf.matmul(output, weights_output_hidden) + bias_output)
        
        memory_cell = tf.tanh(tf.matmul(input, weights_memory_cell) + tf.matmul(output, weights_memory_cell_hidden) + bias_memory_cell)
        
        state = state * forget_gate + input_gate * memory_cell
        
        output = output_gate * tf.tanh(state)
        return state, output
    
    
    outputs = []
    for i in range(batch_size): #Iterates through every window in the batch
        #for each batch I am creating batch_state as all zeros and output for that window which is all zeros at the beginning as well.
        batch_state = np.zeros([1, hidden_layer], dtype=np.float32) 
        batch_output = np.zeros([1, hidden_layer], dtype=np.float32)
        #for each point in the window we are feeding that into LSTM to get next output
        for ii in range(window_size):
            batch_state, batch_output = LSTM_cell(tf.reshape(inputs[i][ii], (-1, 1)), batch_state, batch_output)
        #last output is conisdered and used to get a prediction
        outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)
        
        
    outputs
    
    losses = []
    
    for i in range(len(outputs)):
        losses.append(tf.losses.mean_squared_error(tf.reshape(targets[i], (-1, 1)), outputs[i]))
        
    loss = tf.reduce_mean(losses)
    
    gradients = tf.gradients(loss, tf.trainable_variables())
    clipped, _ = tf.clip_by_global_norm(gradients, clip_margin)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    trained_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    for i in range(epochs):
        traind_scores = []
        ii = 0
        epoch_loss = []
        while(ii + batch_size) <= len(X_train):
            X_batch = X_train[ii:ii+batch_size]
            y_batch = y_train[ii:ii+batch_size]
            
            o, c, _ = session.run([outputs, loss, trained_optimizer], feed_dict={inputs:X_batch, targets:y_batch})
            
            epoch_loss.append(c)
            traind_scores.append(o)
            ii += batch_size
        if (i % 10) == 0:
            print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))
    
    
    
    sup =[]
    for i in range(len(traind_scores)):
        for j in range(len(traind_scores[i])):
            sup.append(traind_scores[i][j][0])
            
    tests = []
    i = 0
    while i+batch_size <= len(X_test):
        
        o = session.run([outputs], feed_dict={inputs:X_test[i:i+batch_size]})
        i += batch_size
        tests.append(o)
        
    
    tests_new = []
    for i in range(len(tests)):
        for j in range(len(tests[i][0])):
            tests_new.append(tests[i][0][j])
            
    test_results = []
    for i in range(661):
        if i >= 556:
            test_results.append(tests_new[i-556])
        else:
            test_results.append(None)
    
    plt.figure(figsize=(16, 7))
    plt.plot(scaled_data, label='Original data')
    plt.plot(sup, label='Training data')
    plt.plot(test_results, label='Testing data')
    plt.legend()
    plt.show()
    
    inv_ytrain = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
    inv_test = scaler.inverse_transform(np.array(tests_new).reshape(-1, 1))
    
    residuals = inv_test - inv_ytrain
    rmse = np.sqrt(np.mean(residuals ** 2))
    R2 =1-((residuals.transpose()@residuals)/(np.var(inv_ytrain)*len(sup)))
    print('RMSE is: ', rmse)   
    session.close()
    
    #indata = scaler.inverse_transform(test_results)
    
    return R2, inv_test, inv_ytrain

Rsq, pred_values, test_values = lstm_tf(scaled_data)

#print( "R Squared for LSTM Model: ", Rsq)