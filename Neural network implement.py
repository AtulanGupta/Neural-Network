# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:59:46 2022

@author: Atulan
"""
# to visualize we can use the following library
# 1. matplotlib- for general visualise
# 2. seaborn- mostly used
# 3. pyplot- interactive visualisation 
# 4. bokeh-    "             "
# 5. d3.js- live data
from sklearn.preprocessing import MinMaxScaler # Random data set is not properly distributed. So we use various
                                               #  scaling process like MinMaxScaler to bring the data within a range.
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

%matplotlib inline
# 1.  Input Data
# 2.  OutPut Data
# 3.  Predicted Result/Data
# 4.  Test data


#            Train Data
input_train_data = np.array([[0, 1, 0],
                           [0, 1, 1], 
                           [0, 0, 0],
                           [10, 0, 0], 
                           [10, 1, 1], 
                           [10, 0, 1]])

out_train_data = np.array([[0], [0], [0], [1], [1], [1]]) # output train data can be of single dimension array or multi 
                                                          # dimension array. Here we are using a single dimensional array.

input_data_prediction = np.array([1, 1, 0]) # This prediction array is created based on input train data.



#         Test Data

input_test_data = np.array([[1, 1, 1],
                            [10, 0, 1], 
                            [0, 1, 10],
                            [10, 1, 10], 
                            [0, 0, 0], 
                            [0, 1, 1]])
 
out_test_data = np.array([[0], [0], [0], [1], [1], [1]])
scaller = MinMaxScaler()
input_train_data_scaled = scaller.fit_transform(input_train_data)
input_test_data_scaled = scaller.fit_transform(input_test_data)

out_train_data_scaled = scaller.fit_transform(out_train_data)
out_test_data_scaled = scaller.fit_transform(out_test_data)
print("Input TrainScalled Data: ", input_train_data_scaled)
print("\n Input Test Scalled Data: ", input_test_data_scaled )

print("\n Output Train Scalled Data: ", out_train_data_scaled)
print("\nOutput test Scalled Data: ", out_test_data_scaled)
#.npy
'''
np.save("Save data/input_train_data_scaled.npy", input_train_data_scaled)
np.save("Save data/input_test_data_scaled.npy",input_test_data_scaled)
np.save("Save data/input_data_prediction.npy",input_data_prediction)
np.save("Save data/out_train_data_scaled.npy", out_train_data_scaled)
np.save("Save data/out_test_data_scaled.npy", out_test_data_scaled)
class Neuralnetwork():
    
    def __init__(self):                             # default constructor
        
        self.inputSize = 3
        self.hiddenSize = 3
        self.outputSize = 1
        
        
        self.W1 = np.random.rand(self.inputSize, self.hiddenSize) # creates a 3X3 matrix
                                                                  # random.rand- positive random float
                                                                  # random.randint- positive random integer
                                                                  # random. randn- positive and negative float
                                                                  # random.random- positive random float 
        self.W2 = np.random.rand(self.hiddenSize, self.outputSize)  
        self.limit = 0.5    # threshold value for error
        # w1- weight between input and hidden layer, w2- weight between hidden layer and output
        
        
        self.error_list = []
        
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0   
        
    def forward(self, X):       # output= weight * input node 
        self.z = np.matmul(X, self.W1)     # self is creating a reference to call on the global variable
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o
        
    def sigmoid(self, s):
        return 1 / (1+ np.exp(-s))
        
    
    def sigmoidPrime(self, s):
        return s*(1-s)
    
    
    
    def backward(self, X, y, o):       # error calculation and weight modification
        self.o_error = y - o    # y- dependent variable(test data) o- output value 
        self.o_delta = self.o_error *  self.sigmoidPrime(o)
        
        self.z2_error = np.matmul(self.o_delta, np.matrix.transpose(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2_error)
        
        self.W1 += np.matmul(np.matrix.transpose(X),self.z2_delta)
        self.W2 += np.matmul(np.matrix.transpose(self.z), self.o_delta)
        
        
    def train(self, X, y, epochs):      # run epoch to get output and error
        for epoch in range(epochs):
            o = self.forward(X)
            self.backward(X, y, o)
            self.error_list.append(np.abs(self.o_error).mean()) # see line number 15

    def predict(self,x_test_data):
        return self.forward(x_test_data).item()
    
        
    
    def error_visualization(self):      # plotting error in graph
        
        plt.plot(range(len(self.error_list)), self.error_list)
        plt.title("Error Visulatization")
        plt.xlabel("epochs")
        plt.ylabel("Loss Value")
        
        
        
    def evaluate(self, input_Data, output_Data):
        for i , testingEvaluation in enumerate(input_Data):
            
            if self.predict(testingEvaluation) > self.limit and output_Data[i] == 1:
                self.true_positives +=1
            if self.predict(testingEvaluation) < self.limit and output_Data[i] == 1:
                self.true_negatives +=1
            if self.predict(testingEvaluation) > self.limit and output_Data[i] == 0:
                self.false_positives += 1
            if self.predict(testingEvaluation) < self.limit and output_Data[i] == 0:
                self.false_negatives +=1
                
                
        print("True Positive Values:", self.true_positives)
        print("True negatives Values:", self.true_negatives)
        print("False Positive Values:", self.false_positives)
        print("False negatives Values", self.false_negatives)
        
        

        
    #.npy

train_scalled_data = np.load("Save data/input_train_data_scaled.npy")
test_scalled_data = np.load("Save data/input_test_data_scaled.npy")
prediction_scalled_data = np.load("Save data/input_data_prediction.npy")

output_scalled_data = np.load("Save data/out_train_data_scaled.npy")
output_testscalled_data = np.load("Save data/out_test_data_scaled.npy")


np.save("Save data/input_train_data_scaled.npy", input_train_data_scaled)
np.save("Save data/input_test_data_scaled.npy",input_test_data_scaled)
np.save("Save data/input_data_prediction.npy",input_data_prediction)
np.save("Save data/out_train_data_scaled.npy", out_train_data_scaled)
np.save("Save data/out_test_data_scaled.npy", out_test_data_scaled)

       
Network = Neuralnetwork()
Network.train(input_train_data_scaled, output_scalled_data, 100)
Network.predict(prediction_scalled_data)
Network.error_visualization()
Network.evaluate(input_test_data_scaled, out_test_data_scaled)
        
import numpy as np
from tensorflow.keras.models import Sequential        
from tensorflow.keras.layers import Dense
trainingData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], 'float64') 
TestingData = np.array([[0], [1], [1], [0]], 'float64')
print(trainingData.shape)
print(TestingData.shape)
model = Sequential()            # we r designing a Fully connected network. Sequential mainly provides an empty space for our network.                      
# # First Layer 

# model.addd(Conv2D())
# model.add(MaxPooling2D)
# model.Dropout()

# # Hidden Layer 01
# model.addd(Conv2D())
# model.add(MaxPooling2D)
# model.Dropout()

# # Hidden Layer 02

# model.addd(Conv2D())
# model.add(MaxPooling2D)
# model.Dropout()


# # Output Layer or Connected Layer 01
# FCNN

model.add(Dense(32, input_dim = 2, activation = 'relu'))      # for continuous data we use Relu activation func, 32- number of node or num of neuron at input 
model.add(Dense(1, activation = 'sigmoid'))                   # for hidden layer design, 


model.compile(loss = 'mean_squared_error',          # activation func sigmoid hoy tahole loss fuction hobe mean-squared-error function; onno gula use hobe na
            optimizer = 'adam',                     # most common optimizer= adam
            metrics = ['binary_accuracy'])

model.fit(trainingData, TestingData, epochs = 100)    # number of data onek beshi hole like 1 million hole epoch barabo, na hoy epoch 20, 30 holei valo.
result = model.evaluate(trainingData, TestingData)
model.predict(trainingData.round( ))    # checking accuracy
testingData = np.array([[0], [1], [1], [0]], 'float64')'''
        