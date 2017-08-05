from __future__ import print_function

import numpy as np

# Confusion matrix representation:
# 		0 | TN | FP |
# Truth	  |---------|
#		1 | FN | TP |
#		  |---------|
#            0   1
#          Predicted

class ConfusionMatrix:
	def __init__ (self):
		self.matrix = np.ndarray((4, 4), dtype=np.uint8)
		self.reset_metrics();

	def append_sample (self, truth, predicted):
		if truth not in [0, 1] or predicted not in [0, 1]:
			raise ValueError('Ground thruth or predicted values must be 0 or 1')

		matrix[truth, predicted] += 1

	def reset_metrics(self):
		self.matrix.fill(0)

	def get_metrics(self):
		TP = float(matrix[1, 1])
		FN = float(matrix[1, 0])
		FP = float(matrix[0, 1])

		recall 		= TP / (TP + FN)
		precision 	= TP / (TP + FP)

		return recall, precision


def test():
	cmatrix = ConfusionMatrix()



if __name__ == '__main__':
    test()
