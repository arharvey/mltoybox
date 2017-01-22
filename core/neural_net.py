import ops

import numpy as np


#------------------------------------------------------------------------------

class Layer(object):
    '''A layer of neurons'''

    def __init__(self, op, size):
        self._op = op
        self._size = size


    @property
    def size(self):
        return self._size


    @property
    def op(self):
        return self._op


#------------------------------------------------------------------------------

class EvaluationResults:

    def __init__(self):
        self.clear()


    def clear(self):
        self.X = []


#------------------------------------------------------------------------------

class NeuralNet(object):
    '''A deep neural network'''

    def __init__(self, num_inputs):
        self._layers = []
        self._num_inputs = num_inputs

        # Weights
        self._W = []


    def addLayer(self, layer):
        '''Adds layer onto end of network and return it's index'''

        index = len(self._layers)

        self._layers.append(layer)

        return index


    def numInputs(self, layer_index):
        return self._layers[layer_index-1].size if layer_index > 0 else self._num_inputs


    def numOutputs(self, layer_index):
        return self._layers[layer_index].size


    def initWeights(self, seed):

        # Ensure that our initial weights are repeatable
        np.random.seed(seed)

        self._W = []

        for layer_index, layer in enumerate( self._layers ):

            # Columns of our weight matrix relate to inputs, rows are our outputs
            W_shape = [layer.size, self.numInputs(layer_index)+1]

            self._W.append( np.random.random(W_shape) )


    def printWeights(self):

        for index, W in enumerate( self._W ):
            print 'Weights for Layer {0}:'.format( index )
            print W
            print


    def evaluate(self, x, results):
        '''Evaluate network for input vector X'''

        results.clear()
        results.X.append(x)

        # Step through layers in order, from front to back
        y = x
        for layer_index, layer in enumerate(self._layers):
            y = layer.op.evaluate( y, self._W[layer_index] )

            # Cache intermediate results. We need these for efficient backprop
            results.X.append(y)
        
        return y


    def evaluateBatch(self, X):

        results = EvaluationResults()

        fn = lambda a: self.evaluate(a, results)

        Y = np.apply_along_axis( fn, 1, X )
        return Y


    def calculateCost(self, T, Y, regularisation):
        '''Returns cost for the training set T'''

        # Number of training examples
        m = T.shape[0]

        cost = -1.0/m * ( np.sum( np.multiply( T, np.log(Y) ) + np.multiply( 1.0-T, np.log(1.0-Y)) ) )

        # Add regularisation term (we ignore weights relating to bias)
        for W in self._W:
            cost += regularisation/(2.0*m) * np.sum( np.multiply( W[:,1:], W[:,1:]) )

        return cost

    def backprop(self, dE_dy, dW, results):
        
        # Process layers back to front
        for layer_index, layer in reversed( list(enumerate(self._layers)) ):

            x = results.X[layer_index]
            y = results.X[layer_index+1]
            W = self._W[layer_index]

            dE_dx, dE_dw = layer.op.backprop(x, y, W, dE_dy)

            dW[layer_index] += dE_dw

            # First element is how much the error varies with changes in the
            # x_bias. However, x_bias is hardcoded to 1.0 and cannot change. We
            # therefore throw this redundant element away. (Also necessary for
            # all the numpy matrix and vector shapes to align correctly!)
            dE_dy = dE_dx[1:]
            
        return dE_dy


    def train(self, X, T, regularisation, learning_rate):
        results = EvaluationResults()

        # Number of training examples
        m = X.shape[0]

        # Prepare to accumulate weight deltas
        dW = [ np.zeros(W.shape) for W in self._W ] 

        # For every training example, do backprop
        for x, t in zip(X, T):
            y = self.evaluate(x, results)

            dE_dy = -(t-y)
            self.backprop( dE_dy, dW, results)

        # Finalise weight deltas
        for layer_index, dW_l in enumerate(dW):
            dW_l *= 1.0/m

            # Add regularisation term (for all weights except bias values)
            dW_l[:,1:] += regularisation * self._W[layer_index][:,1:]

        # Nope... update weights
        for layer_index, W in enumerate(self._W):
            W -= learning_rate * dW[layer_index]


    



