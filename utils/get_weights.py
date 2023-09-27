import numpy as np
import scipy
import scipy.stats as stats
import torch

def weights(num_chunks, max_chunk = 30):
	std = 0.5

	frame = torch.zeros((max_chunk, 5)).float()
	output = []
	for i in range(num_chunks):
		mu = (i*4)/(num_chunks-1)
		l = stats.norm.pdf(range(5), mu, std)
		m1 = min(l)
		m2 = max(l)
		l = [(x-m1)/(m2-m1) for x in l]
		output += [l]
		# print(l)

	output = torch.Tensor(output).float()
	frame[:num_chunks] = output
	return frame

if __name__ == "__main__":
	print(weights(30))