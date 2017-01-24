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

def main():
    print "Loading training data..."
    dataset = training_data.load()

    dims = dataset['rows_cols']

    training_images = dataset['training']['images']
    training_labels = dataset['training']['labels']
    num_training = training_images.shape[0]
    
    print "Found", num_training, 'training examples'

    num_display = 80

    rnd_indices = np.array( np.random.rand(num_display) * num_training, dtype=np.int)

    #displayImages(training_images[rnd_indices], dims)

    num_inputs = training_images.shape[1]
    num_outputs = training_labels.shape[1]

    random_weights_seed = 0xab5a32c0

    net = nn.NeuralNet(num_inputs)

    net.addLayer( nn.Layer(ops.ReLUOp(), num_inputs) )
    #net.addLayer( nn.Layer(ops.ReLUOp(), num_inputs) )
    net.addLayer( nn.Layer(ops.SigmoidOp(), num_outputs) )

    net.initWeights(random_weights_seed)

    params = nn.TrainingParameters()
    params.learning_rate = 1.0
    params.regularisation = 0.0
    params.batch_size = 10
    params.report_cost_iteration_count = 1

    sel = np.array( np.random.rand(100) * num_training, dtype=np.int)


    report = net.miniBatchTraining(training_images[sel], training_labels[sel], params)
    print report
    
    net.printWeights()
    vis.plotReport(report)

if __name__ == "__main__":
    main()



