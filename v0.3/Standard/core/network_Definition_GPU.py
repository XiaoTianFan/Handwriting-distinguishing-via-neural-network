# compute by numpy & cupy
import numpy  
import cupy
# scipy.special for the sigmoid function expit(), and its inverse logit()
import scipy.special
import os
import time


# neural network class definition
class neuralNetwork:
    

    # initialise the neural network
    def __init__(self, inputnodes=784, hiddennodes=0, outputnodes=0, learningrate=0, reload_net=False):
        # when there's given no previous Net 
        if reload_net == False:
            # set number of nodes in each input, hidden, output layer
            self.inodes = inputnodes
            self.hnodes = hiddennodes
            self.onodes = outputnodes
            
            # link weight matrices, wih and who
            # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
            # w11 w21
            # w12 w22 etc 
            self.wih = cupy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
            self.who = cupy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

            # learning rate
            self.lr = learningrate

        # when there's given a previous Net 
        elif reload_net != False:
            # read inital settings file
            with open('../save/' + reload_net + '/initial_setting.txt', 'r') as read_file:
                net_setting = read_file.readlines()
            
            # set number of nodes in each input, hidden, output layer and learning rate from previous settings
            self.inodes = int(net_setting[0][:2])
            self.hnodes = int(net_setting[1][:2])
            self.onodes = int(net_setting[2][:2])
            self.lr = float(net_setting[3][:2])
            
            # set weights from previous net
            self.wih = cupy.array(numpy.loadtxt(open('../save/' + reload_net + '/wih.csv',"rb"), delimiter=",", skiprows=0))
            self.who = cupy.array(numpy.loadtxt(open('../save/' + reload_net + '/who.csv',"rb"), delimiter=",", skiprows=0))

        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)    
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = cupy.array(inputs_list, ndmin=2).T
        targets = cupy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = cupy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = cupy.array(self.activation_function(cupy.ndarray.get(hidden_inputs))) #to activate sigmoid, invert cupy.ndarray to numpy object temporarily
        
        # calculate signals into final output layer
        final_inputs = cupy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = cupy.array(self.activation_function(cupy.ndarray.get(final_inputs)))
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = cupy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * cupy.dot((output_errors * final_outputs * (1.0 - final_outputs)), cupy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * cupy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), cupy.transpose(inputs))
        
        return output_errors

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = cupy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = cupy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = cupy.array(self.activation_function(cupy.ndarray.get(hidden_inputs)))
        
        # calculate signals into final output layer
        final_inputs = cupy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = cupy.array(self.activation_function(cupy.ndarray.get(final_inputs)))
        
        return final_outputs
    
    
    # backquery the neural network
    # use the same termnimology to each item, 
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = cupy.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = cupy.array(self.inverse_activation_function(cupy.ndarray.get(final_outputs)))

        # calculate the signal out of the hidden layer
        hidden_outputs = cupy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= cupy.min(hidden_outputs)
        hidden_outputs /= cupy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hidden layer
        hidden_inputs = cupy.array(self.inverse_activation_function(cupy.ndarray.get(hidden_outputs)))
        
        # calculate the signal out of the input layer
        inputs = cupy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= cupy.min(inputs)
        inputs /= cupy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs

    # save the Network in fold
    def saveNetwork(self):
        # create fold and file
        fold_name = '../save/Network '+str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        os.makedirs(fold_name) 
        file_name = fold_name+'/initial_setting.txt'

        # save initial settings of Network
        initial_data = [str(self.inodes)+'\n', str(self.hnodes)+'\n', str(self.onodes)+'\n', str(self.lr)+'\n', ]
        with open(file_name, 'w') as newSaveNetwork:
            newSaveNetwork.writelines(initial_data)

        # save current weights individually
        numpy.savetxt(fold_name+"/wih.csv", cupy.ndarray.get(self.wih), delimiter=',')
        numpy.savetxt(fold_name+"/who.csv", cupy.ndarray.get(self.who), delimiter=',')

        pass        
        

