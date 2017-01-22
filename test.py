#!/usr/bin/env python

import core.ops as ops
import core.neural_net as nn
import core.vis as vis

import numpy as np
import matplotlib.pyplot as plt

import math

h = 1e-6

#------------------------------------------------------------------------------

def testLinearOp():

    # Two inputs
    x = np.array( [2.0,3.0] )

    # Weights for two neurons. First weight is for bias
    W = np.array( [ [1.0,10.0,100.0],
                    [2.0, 4.0, 8.0] ] )

    # Number of neurons is implied by the number of rows in the weight matrix
    lin_op = ops.LinearOp()

    y = lin_op.evaluate( x, W )

    assert( (y == np.array([321., 34.0])).all() )


    # Differentiate

    print 'dy_dw (fd) =', ops.differentiateByFiniteDifference(lin_op, x, W, h)
    
    dy_dx, dy_dw = lin_op.backprop( x, y, W, np.ones(W.shape[0]) )
    print 'dy_dx =', dy_dx
    print 'dy_dw =', dy_dw


#------------------------------------------------------------------------------

def testSigmoidOp():

    # Plot output of a single neuron 

    X = np.linspace(-5.0, 5.0, 50)
    Y = np.zeros(X.shape)

    W = np.array([[0.0, 1.0]])

    sig_op = ops.SigmoidOp()

    for i, xx in enumerate(X):
        x = np.array([xx])
        Y[i] = sig_op.evaluate( x, W )

    # Plot
    if False:
        plt.plot( X, Y )
        plt.show()

    # Check derivatives
    for i, xx in enumerate(X):
        x = np.array([xx])
        dy_dw_fd = ops.differentiateByFiniteDifference( sig_op, x, W, h)
        dy_dx, dy_dw = sig_op.backprop(x, Y[i], W, np.ones(W.shape[0]))

        print dy_dw_fd - dy_dw


#------------------------------------------------------------------------------


def testSingleNeuron():

    # Our basic test data

    X0 = np.array([ [0.8, 0.6],
                    [0.9, 0.1] ])

    X1 = np.array([ [0.2, 0.3],
                    [0.1, 0.7] ])
    

    # Display training data

    plt.plot( X0[:,0], X0[:,1], 'ro' )
    plt.plot( X1[:,0], X1[:,1], 'bs' )
    plt.axis( [0.0, 1.0, 0.0, 1.0] )
    plt.show()

    # A trivial neural net

    num_inputs = 2
    random_weights_seed = 0xab5a32c0

    net = nn.NeuralNet(num_inputs)

    net.addLayer( nn.Layer(ops.LinearOp(), 2) )
    net.addLayer( nn.Layer(ops.SigmoidOp(), 1) )

    net.initWeights(random_weights_seed)

    net.printWeights()

    X = np.vstack( (X0, X1 ) )
    T = np.vstack( (np.zeros(X0.shape[0]).reshape(-1,1), 
                    np.ones(X1.shape[0]).reshape(-1,1) ) )

    print 'X = \n{0}'.format(X)
    print 'T = \n{0}'.format(T)

    learning_rate = 4
    regularisation = 0.0

    cost_threshold = 1e-3
    cost_abort_threshold = 1e-8
    max_iterations = 2000
    report_cost_iteration_count = 200

    cost_history = []

    prev_cost = float('inf')
    iterations = 0
    while iterations < max_iterations:

        Y = net.evaluateBatch(X)
        #print 'Y = \n{0}'.format(Y)

        cost = net.calculateCost( T, Y, regularisation )
        cost_history.append( cost )

        if iterations % report_cost_iteration_count == 0:
            print "{0}: Cost {1}".format(iterations, cost)

        # Are we done?
        if cost < cost_threshold:
            print "Completed"
            break

        # Should we give up
        if iterations > 0 and abs(cost - prev_cost) < cost_abort_threshold:
            print "Aborting. Cost has converged"
            break

        prev_cost = cost
        iterations = iterations + 1

        # Perform a round of training
        net.train( X, T, regularisation, learning_rate)

    if iterations == max_iterations:
        print "Aborting. Maximum iterations reached"

    print 'Total iterations:', iterations
    print 'Final cost:', cost
    net.printWeights()

    vis.plotCost( np.array(cost_history), 500)



#------------------------------------------------------------------------------

def main():
    #testLinearOp()
    #testSigmoidOp()
    testSingleNeuron()

if __name__ == "__main__":
    main()



