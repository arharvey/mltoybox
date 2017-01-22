#!/usr/bin/env python

import os
import sys

# Make sure we can find mltoybox modules
root_dir = os.path.abspath( os.path.join( os.path.dirname(__file__), '../../') )
sys.path.append( root_dir )


import core.file_utils as file_utils


# See http://yann.lecun.com/exdb/mnist/ for more information.
#
# "The MNIST database of handwritten digits, available from this page, has a
# training set of 60,000 examples, and a test set of 10,000 examples. It is a
# subset of a larger set available from NIST. The digits have been size-
# normalized and centered in a fixed-size image.
#
# "It is a good database for people who want to try learning techniques and
# pattern recognition methods on real-world data while spending minimal efforts
# on preprocessing and formatting."
#

MNIST_URL = {   'train-images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                'train-labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                'test-images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                'test-labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz' }


def fetch():
    print 'Fetching MNIST data...'

    dest = file_utils.ensureDownloadDirectoryExists(__file__)

    files = file_utils.download( [ url for _, url in MNIST_URL.iteritems() ], dest)
    file_utils.unzip(files, dest)

    print 'Removing temporary files...'
    file_utils.removeFiles( files )


if __name__ == "__main__":
    fetch()
