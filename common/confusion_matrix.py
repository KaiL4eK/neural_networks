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
		matrix = np.ndarray((4, 4), dtype=np.uint8)

	def append_sample (self, truth, predicted):
		if truth not in [0, 1] or predicted not in [0, 1]:
			raise ValueError('Ground thruth or predicted values must be 0 or 1')


