import torch 
import time
import numpy as np
from numba import jit

torch.set_printoptions(threshold = 50000)
np.set_printoptions(threshold = 50000)

@jit
def DynamicProgram(N, T, reward_mat, pre_sum_mat, boundry_mat):
	j_offset = 1
	for i in range(1, T):
		repeat_flag = 0
		for j in range(j_offset, N):
			for k in range(j_offset, j + 1):
				reward_candidate = reward_mat[i - 1, k] + pre_sum_mat[i, j] - pre_sum_mat[i, k]
				if reward_candidate > reward_mat[i, j]:
					reward_mat[i, j] = reward_candidate
					boundry_mat[i, j] = k
			if boundry_mat[i, j] == j and repeat_flag == 0:
				j_offset = int(boundry_mat[i, j])
			elif boundry_mat[i, j] != j :
				repeat_flag += 1
	return boundry_mat


def DurationExtraction(alignment):
	T, N = alignment.shape
	reward_mat = torch.zeros_like(alignment)
	boundry_mat = torch.zeros_like(alignment)
	pre_sum_mat = torch.cumsum(alignment, dim=1)
	reward_mat = reward_mat.cpu().numpy().astype(np.float32)
	boundry_mat = boundry_mat.cpu().numpy().astype(np.int32)
	pre_sum_mat = pre_sum_mat.cpu().numpy().astype(np.float32)
	
	reward_mat[0, :] = pre_sum_mat[0, :]
	boundry_mat = DynamicProgram(N, T, reward_mat, pre_sum_mat, boundry_mat)
	
	p = N - 1
	duration = []
	for i in range(T, 0, -1):	
		duration.append(p - int(boundry_mat[i-1, p].item()))
		p = int(boundry_mat[i-1, p].item())
	duration.reverse()
	return duration

@jit
def fuse_blank(cost_matrix, target_len):
	cost_matrix = np.exp(cost_matrix)
	_, T = cost_matrix.shape
	score_matrix = cost_matrix[1::2]
	for i in range(1, target_len):
		blank_anchor = 2 * i
		for j in range(T):
			if cost_matrix[blank_anchor - 1, j] >= cost_matrix[blank_anchor + 1, j]:
				score_matrix[i - 1, j] = cost_matrix[blank_anchor, j]
			else:
				score_matrix[i, j] = cost_matrix[blank_anchor, j]
	score_matrix[0] = score_matrix[0] + cost_matrix[0]
	score_matrix[-1] = score_matrix[-1] + cost_matrix[-1]

	return score_matrix

@jit
def make_alpha(T, L, log_alpha, log_y, labels):
	for t in range(1, T):
		for i in range(L):
			s = labels[i]
			a = log_alpha[t - 1, i]
			b = log_alpha[t - 1, i - 1]
			c = log_alpha[t - 1, i - 2]
			if i - 1 >= 0:
				if b == -np.inf:
					a = a
				elif a < b:
					a = b + np.log(1 + np.exp(a - b))
				else:
					a = a + np.log(1 + np.exp(b - a)) 
			if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
				if c == -np.inf:
					a = a
				elif a < c:
					a = c + np.log(1 + np.exp(a - c))
				else:
					a = a + np.log(1 + np.exp(c - a)) 
			log_alpha[t, i] = a + log_y[t, s]
	return log_alpha

def forward_log(log_y, labels):
	T, V = log_y.shape
	L = len(labels)
	log_alpha = np.ones([T, L]) * (-np.inf)

	log_alpha[0, 0] = log_y[0, labels[0]]
	log_alpha[0, 1] = log_y[0, labels[1]]

	return make_alpha(T, L, log_alpha, log_y, labels)

def CTCAlignment(log_prob, target, blank_index):
	log_prob, target = log_prob.cpu().numpy(), target.cpu().numpy()
	L = len(target)
	labels = np.array([blank_index] * (2 * L + 1))
	for i in range(L):
		labels[2 * i + 1] = target[i]
	cost_matrix = forward_log(log_prob, labels)
	return fuse_blank(cost_matrix.T, L)

if __name__ == '__main__':
	bz = 16
	n = 6000
	t = 600
	v = 60
	torch.manual_seed(42)
	_scores = torch.randn((bz, n, v), requires_grad=False, device="cuda:0")
	scores = _scores.log_softmax(2)
	targets = torch.randint(0, v, (bz, t), device="cuda:0")
	input_lengths = torch.tensor([n] * bz, device="cuda:0")
	target_lengths = torch.tensor([t] * bz, device="cuda:0")

	start = time.time()
	score_matrix = CTCAlignment(scores[0], targets[0], 0)
	end = time.time()
	print(f'score_matrix {score_matrix} cost time {end-start}')