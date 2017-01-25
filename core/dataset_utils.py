import numpy as np
import math

def splitTrainingData(X, Y, cv_fraction, subset, seed):

    np.random.seed(seed)

    num_training = X.shape[0]

    sel = np.arange(num_training)
    np.random.shuffle(sel)

    if subset is not None:
        sel = sel[:subset]
        num_training = subset
    
    k = int( math.ceil( float(num_training)*(1.0-cv_fraction) ) )

    cv_X = X[ sel[k:] ]
    cv_Y = Y[ sel[k:] ]

    training_X = X[ sel[:k] ]
    training_Y = Y[ sel[:k] ]

    return (training_X, training_Y, cv_X, cv_Y)

