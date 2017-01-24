import numpy as np

import exceptions

def differentiateByFiniteDifference(op, x, W, h):
    '''Differentiates an op using finite differencing. Only for diagnostic purposes'''

    y = op.evaluate( x, W )

    dy_dW = np.zeros( W.shape )

    nx = x.shape[0]
    ny = W.shape[0]

    # Strictly speaking, we should calculate partial derivatives for every
    # combination of y and W elements. However, we assume that each element of y
    # is only affected by a single row in W.
    #

    for i in xrange(nx+1): # x does not include an explicit 1.0 for bias weights
        Wi = np.copy(W)
        v = np.tile(h, ny)
        Wi[:,i] += v
        
        dy_dW[:,i] = (op.evaluate(x, Wi) - y) / h

    return dy_dW


class Op(object):
    '''Base operation. Different types of neuron are derived from this'''


    def evaluate(self, x, W):
        '''Given input vector x and weight matrix W, calculate the output vector y

        x does not include bias term (which has an assumed value of 1.0)'''
        raise exceptions.NotImplementedError


    def backprop(self, x, y, W, dE_dy):
        '''Given signal x, result y, weights W and back propogated dE_dy, calculate dE_dx and dE_dw.

        dE_dx is used for back propogation. dE_dw for learning weights in gradient descent'''
        raise exceptions.NotImplementedError



class LinearOp(Op):
    '''A simple layer of linear neurons'''

    def evaluate(self, x, W):
        # First column of weights are the bias values
        x_with_bias = np.hstack([1.0, x])
        y = np.dot( W, x_with_bias )
        return y


    def backprop(self, x, y, W, dE_dy):
        
        # dE_dx
        dE_dx = np.dot(W.T, dE_dy) 

        # dE_dw
        x_with_bias = np.hstack([1.0, x]) # x_bias is hardcoded to 1.0
        dy_dw = np.tile( x_with_bias, (W.shape[0],1) )
        dE_dw = np.multiply( dy_dw, dE_dy.reshape(-1,1) )

        return (dE_dx, dE_dw)


class ReLUOp(Op):
    '''Rectified linear unit'''

    def evaluate(self, x, W):
        # First column of weights are the bias values
        x_with_bias = np.hstack([1.0, x])
        z = np.dot( W, x_with_bias )
        y = z * (z > 0.0)

        return y


    def backprop(self, x, y, W, dE_dy):
        
        dy_dz = (y > 0.0).astype(float)
        dE_dz = np.multiply(dy_dz, dE_dy)

        # dE_dx
        dE_dx = np.dot(W.T, dE_dz) 
        
        # dE_dw
        x_with_bias = np.hstack([1.0, x]) # x_bias is hardcoded to 1.0

        dz_dw = np.tile( x_with_bias, (W.shape[0],1) )
        dE_dw = np.multiply( dz_dw, dE_dz.reshape(-1,1) )

        return (dE_dx, dE_dw)


class SigmoidOp(Op):

    def evaluate(self, x, W):
        # First column of weights are the bias values
        z = np.dot( W, np.hstack([1.0, x]) )
        y = 1.0 / (1.0 + np.exp(-z))

        return y

    def backprop(self, x, y, W, dE_dy):

        dy_dz = np.multiply(y, (1.0-y))
        dE_dz = np.multiply(dy_dz, dE_dy)

        # dE_dx
        dE_dx = np.dot(W.T, dE_dz) 
        
        # dE_dw
        x_with_bias = np.hstack([1.0, x]) # x_bias is hardcoded to 1.0

        dz_dw = np.tile( x_with_bias, (W.shape[0],1) )
        dE_dw = np.multiply( dz_dw, dE_dz.reshape(-1,1) )
        
        return (dE_dx, dE_dw)










