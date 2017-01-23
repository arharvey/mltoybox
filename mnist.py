#!/usr/bin/env python

import data.mnist.training_data as training_data

import pprint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import math

import numpy as np

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
    dataset = training_data.load()

    dims = dataset['rows_cols']

    training_images = dataset['training']['images']
    num_training = training_images.shape[0]
    
    num_display = 80

    rnd_indices = np.array( np.random.rand(num_display) * num_training, dtype=np.int)

    displayImages(training_images[rnd_indices], dims)

if __name__ == "__main__":
    main()



