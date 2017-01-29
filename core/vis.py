import numpy as np
import matplotlib.pyplot as plt

def plotCost(samples, max_samples):

	plt.close()

	num_samples = samples.size

	X = None
	Y = None

	# If we have exceeded the maximum number of samples, only visualise an evenly distributed subset
	if num_samples > max_samples:
		X = np.zeros( max_samples )
		Y = np.zeros( max_samples )

		step = float(num_samples) / max_samples
		for j in xrange(1, max_samples):
			i = int(j * step)

			X[j] = float(i)
			Y[j] = samples[i]

	else:
		X = np.arange( 1, num_samples )
		Y = samples[1:]

	plt.plot(X, Y)
	plt.axis([1, X[-1], 0.0, np.max(Y)])

	plt.show()


def plotReport(report):
	plotCost( report.cost_history, 500)