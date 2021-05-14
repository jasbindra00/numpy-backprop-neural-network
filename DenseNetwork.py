import random
import numpy as np
import math


class DenseNetwork:
    def __init__(self):
        pass

    def initialiseNetwork(self, network_dimensions, weights = None, biases = None):

        #Initialise the weights and biases with the dimensions
        self.network_dimensions = network_dimensions
        self.initialiseWeights(network_dimensions, weights)
        self.initialiseBiases(network_dimensions, biases)


        self.n_layers = len(network_dimensions)
    

    def initialiseWeights(self, network_dimensions, weights):
        #Initialise the weights with a Gaussian distribution with mean of 0 and standard deviation of 1.
            #The weights of a layer are a matrix, since each neuron in the layer is densely connected to the neurons in the previous layer.
                #The jth row represents the weights between the jth neuron and every single neuron in the previous layer.

        self.weights = weights if weights is not None else [np.random.normal(0, 1, (columns, rows)) for rows, columns in zip(network_dimensions[:-1], network_dimensions[1:])]
      


    
    def initialiseBiases(self, network_dimensions, biases):
        #Initialise the biases with a Gaussian distribution with mean of 0 and a standard dev of 1.
        #The biases of a given layer is simply a column vector, where the ith row entry represents the bias of the ith neuron in the layer.

        self.biases = biases if biases is not None else [np.random.normal(0,1, (rows, 1)) for rows in network_dimensions[1:]]
  



    def untrackedFeedforward(self, example): # FINE
        #Compute the output activation by feeding the example into the network.
            #(do not store the activations as this is for testing only)

        for layer_biases, layer_weights in zip(self.biases, self.weights):
            example = sigmoid(np.add(np.matmul(layer_weights, example),layer_biases))
        return example

    
    def trackedFeedforward(self, input): #FINE
        #Forward pass a given input through the neural network.
        
        #As well as the raw activations, as both are used in the backpropagation algorithm.
        raw_layer_activations = [] 

        #We need to keep track of the new activations per layer.
        layer_activations = [input] 


        
        #Cascade through each layer and compute the new activations.
        for current_layer_biases, current_layer_weights in zip(self.biases, self.weights):

            #Raw activation = w(CL)*a(PL)+b(CL)
            raw_layer_activations.append(np.add(np.matmul(current_layer_weights, layer_activations[-1]),current_layer_biases))

            #Pass the raw activation into the activation function
            layer_activations.append(sigmoid(raw_layer_activations[-1]))

        #Return a tuple of (raw activations, activations)
        return(raw_layer_activations, layer_activations)

    def backpropagation(self, raw_activations, activations, label):

        #Backpropagation algorithm used to compute the errors for each layer.
            #(by this point, we have fed our example into the network)
        #1) Compute the output error.
        #2) Find the error of the ith layer by using the error of the (i+1)th layer.

    
        
        def computeOutputError(last_layer_raw_activation, last_layer_activation, label):
            #Helper function to compute 
            #After feeding the example forward into the neural network, we now begin the backpropagation algorithm.
                #We need to find the error of the output, which was proved to be:
            
            #  Hadamard product        Loss with respect to activation                                Rate of change of activation                              
            return np.multiply(self.MSEActivationDerivative(last_layer_activation, label) , sigmoidDerivative(last_layer_raw_activation))

        def computeLayerError(current_layer_raw_activations,next_layer_weights, next_layer_error):
            #Helper function for back propagation, defined by a recurrence relation.
                #The error of the ith layer is a function of the (i+1)th layer.
                
            #By this point, we have computed the output layer error, and thus we can compute the errors for all layers by the recurrence relation.

            #We proved the error of the ith layer as:

            #       Hadamard product                       Transposition of the next layer weights      (matrix multiplication)    The error of the next layer        #Hadamard product with the rate of change of the raw activation
            return     np.multiply(     np.matmul               (next_layer_weights.transpose(),                                           next_layer_error) ,              sigmoidDerivative(current_layer_raw_activations))


        
        #Initialse the layer errors array with the output error.

        layer_errors = [computeOutputError(raw_activations[-1], activations[-1], label)]

        #Iterate through the layers backwards to find the error of each layer and store it in the above array.
        
        for l in range(2, self.n_layers):
            
            layer_errors.insert(0, computeLayerError(raw_activations[-l], self.weights[-l+1], layer_errors[0]))
        #Layer errors is an array [layer_error_1, layer_error_2,...,output_layer_error]
        return layer_errors



    def computeLayerGradients(self, layer_errors, layer_activations):
        #At this point, we have applied backpropagation to compute the layer errors for each layer.
        #We now need to find the gradients of the cost function for the weights and biases of each layer as a result of feeding our sample into the network.

        #We have proved that the gradient of the (i+1)th layer is a function of the activations of the previous layer.

        
        def computeLayerGradient(layer_error, previous_layer_activations):
            #Return a tuple of    Gradient of cost function wrt bias,                  Gradient of cost function wrt weights
            return                   (layer_error,                           np.matmul(layer_error, previous_layer_activations.transpose()))
        
        #Returns an array of the tuples of the gradients for each layer.
        res = [computeLayerGradient(layer_errors[-l], layer_activations[-l - 1]) for l in range(1, self.n_layers)]
        
        #Reverse the list since list comprehension only allows me to construct the gradients of each layer in reverse.
            #Return an array of [(layer_1_bias_gradient, layer_1_weight_gradient),....,(layer_n_bias_gradient, layer_n_weight_gradient)]
        return res[::-1]


    
    def stochasticGradientDescent(self, mini_batch_size, accrued_change_in_biases, accrued_change_in_weights):
        #It's time to apply the change in weights and change in biases.
            #The accrued change in weights are the change in weights for each layer over (mini_batch_size) samples. 
            #We need to average this across each sample and apply the learning rate, which will be the change in weights and bias for each layer for this minibatch.
        
        #This is a constant. 
            #1/mini_batch_size being the average across the accrued matrices.
            #Learning rate being the proportion by which we should step in the cost function wrt to our weights or biases.
        learning_rate_over_minibatch_size = (self.learning_rate / mini_batch_size)

        #Iterate through each layer and apply the changes to the weights and biases.
        for i in range(0, self.n_layers - 1):
     
            #                            Current weight   -   Average change in weights across minibatch
            self.weights[i] = np.subtract(self.weights[i], learning_rate_over_minibatch_size * accrued_change_in_weights[i]) 
            #                            Current bias     -   Average change in biases across minibatch
            self.biases[i] = np.subtract(self.biases[i], learning_rate_over_minibatch_size * accrued_change_in_biases[i]) 




    def train(self, training_data, epochs, mini_batch_size, starting_learning_rate):
        
        #Store the learning rate, as it is used by other functions.
        self.learning_rate = starting_learning_rate
        

        #A single epoch is a pass of the entire training set, once.
        for epoch in range(epochs):

            #Produce the minibatches and process each one.
         
            for mini_batch in DenseNetwork.generateMinibatches(training_data, mini_batch_size):
                self.processMinibatch(mini_batch)
            #Store the accuracy.




      
 
        


    
    #Generate minibatches from an array of (example, label) with size of minibatch_size
    def generateMinibatches(paired_data, minibatch_size):
        #Firstly shuffle the data around.
        random.shuffle(paired_data)

        #Chunk the data up into 'minibatch_size' chunks.
        return [paired_data[i:i+minibatch_size] for i in range(0, len(paired_data), minibatch_size)]

    

    def processMinibatch(self, mini_batch):

        #We want to estimate the total change in the weights and biases by computing the average change in gradients for both weights and biases across every sample in the minibatch.

        #Keep zero matrices of both weights and biases to accrue such changes.
        accrued_network_change_in_biases = [np.zeros(layer_biases.shape) for layer_biases in self.biases]
      
        accrued_network_change_in_weights = [np.zeros(layer_weights.shape) for layer_weights in self.weights]

        #Iterate through each example in the minibatch
        for example, label in mini_batch:
        
            
            #Feed the example into the neural network and get the raw activations and activations of each layer as the result of feeding such an example forward.
            raw_activations, activations = self.trackedFeedforward(example)

            #READ THE BELOW LINE BACKWARDS 
            #Array of tuples of (layer_i_bias_gradient,layer_i_weight_gradient)     #Use the layer errors to compute the new gradients of each layer              (Commence the computation of each layer error through backpropagation.
            layer_gradients                 =                                                            self.computeLayerGradients(                           self.backpropagation(raw_activations, activations, label),           activations)


            #Accrue the gradient for the bias for this given sample, for each layer

            accrued_network_change_in_biases = [np.add(current_layer_bias, layer_gradient[0]) for current_layer_bias, layer_gradient in zip(accrued_network_change_in_biases, layer_gradients)]

            #Accrue the gradient for the weight for this given sample, for each layer
            accrued_network_change_in_weights = [np.add(current_layer_weight, layer_gradient[1]) for current_layer_weight, layer_gradient in zip(accrued_network_change_in_weights, layer_gradients)]
          

        #Now that we have our total change in cost function wrt weights and biases, for each layer, we can apply the change to each layer by averaging across the minibatch.
        self.stochasticGradientDescent(len(mini_batch), accrued_network_change_in_biases, accrued_network_change_in_weights)




    def findAccuracy(self, data):
        if data is None: return None
        res = [(np.argmax(self.untrackedFeedforward(input)), np.argmax(output)) for (input,output) in data]
        return sum(int(input == output) for (input, output) in res) / len(res)

    def predict(self, data):
        #Data is a matrix of column vectors.
        res = [np.reshape(example, (self.network_dimensions[0],1)) for example in data]
        return [np.argmax(self.untrackedFeedforward(example)) for example in res]
       

    def MSEActivationDerivative(self, output_activations, y):
        #The derivative of the MSE cost function wrt activation is simply the following:
        return (output_activations-y)

    def setWeights(self, weights):
        self.weights = weights
    
    def setBiases(self,biases):
        self.biases = biases


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoidDerivative(z):
    return sigmoid(z)*(1-sigmoid(z))

