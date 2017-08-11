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
		truth 		= int(truth)
		predicted 	= int(predicted)

		if truth not in [0, 1] or predicted not in [0, 1]:
			raise ValueError('Ground thruth or predicted values must be 0 or 1')

		self.matrix[truth, predicted] += 1

	def reset_metrics(self):
		self.matrix.fill(0)

	def get_metrics(self):
		TP = float(self.matrix[1, 1])
		FN = float(self.matrix[1, 0])
		FP = float(self.matrix[0, 1])

		recall 		= TP / (TP + FN)
		precision 	= TP / (TP + FP)

		return recall, precision

	def get_f1_metric(self):
		recall, precision = self.get_metrics()

		return 2 * (precision * recall / (precision + recall))


def test():
	cmatrix = ConfusionMatrix()

	truth_samples 		= [ 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1 ]
	predicted_samples 	= [ 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0 ]

	for sample in zip(truth_samples, predicted_samples):
		cmatrix.append_sample(truth=sample[0], predicted=sample[1])
		print('Appending {}'.format(sample))

		if sample[1] == 1:
			if sample[0] == 1:
				print('Result: TP')
			else:
				print('Result: FP')
		else:
			if sample[0] == 0:
				print('Result: TN')
			else:
				print('Result: FN')


	recall, precision = cmatrix.get_metrics()
	print('Recall: {} / Precision: {}'.format(recall, precision))

if __name__ == '__main__':
    test()
