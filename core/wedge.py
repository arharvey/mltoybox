import numpy as np
import matplotlib.pyplot as plt

import core.neural_net as nn

def wedge(fn_create_network, fn_create_params, wedged_values,
            training_data, cv_data, test_data):

    cv_cost_history = []

    for v in wedged_values:

        print "Wedge value:", v

        net = fn_create_network(v)

        params = fn_create_params(v)
        report = net.miniBatchTraining(training_data[0], training_data[1], params)
        
        print report

        # Calculate validation cost

        cv_Y = net.evaluateBatch(cv_data[0])
        cv_cost = net.calculateCost(cv_data[1], cv_Y, params.regularisation)

        print 'Cross-validation cost:', cv_cost

        cv_cost_history.append( cv_cost )

        # Calculate cost of test set
        

        test_Y = net.evaluateBatch(test_data[0])
        test_cost = net.calculateCost(test_data[1], test_Y, params.regularisation)

        print 'Test cost:',test_cost

        hits, num_test = nn.countHits(test_data[1], test_Y)
        print 'Hit rate:', float(hits)/num_test
        
        print


    # Find best wedge value
    cv_cost_history = np.array(cv_cost_history)
    min_cv_cost_index = np.argmin( cv_cost_history )

    print "Best:", wedged_values[min_cv_cost_index], "with cost", cv_cost_history[min_cv_cost_index]

    plt.plot( wedged_values, cv_cost_history )
    plt.show()
