#!/usr/bin/env python

import data.mnist.training_data as training_data

import pprint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import math

import numpy as np

import core.neural_net as nn
import core.vis as vis
import core.ops as ops
import core.dataset_utils as dataset_utils
import core.wedge as wedge

#------------------------------------------------------------------------------

def displayImages(images, dim):

    num_images = images.shape[0]

    # Arrange images in a roughly square grid
    cols = int( math.ceil(math.sqrt(num_images)) )
    rows = int( math.ceil( float(num_images) / cols) )

    gs = gridspec.GridSpec(rows, cols, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
    
    for i in range(rows):
        for j in range(cols):

            idx = i*cols + j
            if idx >= num_images:
                break

            g = gs[i,j]

            img = images[idx].copy().reshape(dim)
            
            ax = plt.subplot(g)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()



#------------------------------------------------------------------------------


def logrange(start, stop, steps):

    ratio = stop/start
    base = ratio ** 1.0/steps

    return np.logspace(0.0, math.log(ratio)/math.log(base), steps, base = base) * start



def create_params(v):
    params = nn.TrainingParameters()
    params.learning_rate = 1
    params.regularisation = 0.0025
    params.batch_size = 10
    params.report_cost_iteration_count = 10
    params.max_iterations = 100

    return params


def create_network(num_inputs, num_outputs, v):

    net = nn.NeuralNet(num_inputs)

    net.addLayer( nn.Layer(ops.ReLUOp(), 280) )
    net.addLayer( nn.Layer(ops.ReLUOp(), v) )
    #net.addLayer( nn.Layer(ops.ReLUOp(), 100) )
    net.addLayer( nn.Layer(ops.SigmoidOp(), num_outputs) )

    random_weights_seed = 0xab5a32c0
    net.initWeights(random_weights_seed)

    return net


def main():
    print "Loading training data..."
    dataset = training_data.load()

    dims = dataset['rows_cols']

    training_images = dataset['training']['images']
    training_labels = dataset['training']['labels']
    num_training = training_images.shape[0]
    
    print "Found", num_training, 'training examples'

    #num_display = 80
    #rnd_indices = np.array( np.random.rand(num_display) * num_training, dtype=np.int)
    #displayImages(training_images[rnd_indices], dims)
    
    # Split training set into training and cross-validation sets

    random_dataset_seed = 0xa85ca2c9

    training_subset = 20000
    cv_fraction = 0.25

    training, cv = dataset_utils.splitTrainingData(training_images, training_labels,
                                        cv_fraction, training_subset,
                                        random_dataset_seed)

    num_training = training[0].shape[0]
    num_cv = cv[0].shape[0]

    print 'Splitting training/cross-validation examples: {0}/{1}'.format(num_training, num_cv)

    num_inputs = training[0].shape[1]
    num_outputs = training[1].shape[1]

    test = ( dataset['test']['images'], dataset['test']['labels'] )

    # Wedge hyper parameters

    print

    wedge_values = logrange(20, 400, 10).astype(int)
    print wedge_values

    wedge.wedge( lambda v: create_network(num_inputs, num_outputs, v),
                 create_params,
                 wedge_values,
                 training, cv, test )


    #vis.plotReport(report)

if __name__ == "__main__":
    main()



