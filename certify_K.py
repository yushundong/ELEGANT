import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from decimal import Decimal
from numba import jit, njit
from numba.typed import List
from time import time
from multiprocessing import Pool


def certify_K(args):
	
	def test_v(p_l, alpha, beta, global_d, v, fn):
		plower_Z = int(p_l * 100 ** 10) * (100 ** (global_d-10))
		pupper_Z = int((1-p_l) * 100 ** 10) * (100 ** (global_d-10))
		total_Z = 100 ** global_d

		complete_cnt = []
		cnt = np.load('list_counts/{}/complete_count_{}.npy'.format(fn, v), allow_pickle=True)
		complete_cnt += list(cnt)

		outcome = []
		for ((s, t), c) in complete_cnt:
			outcome.append(((alpha / beta, t - s), c, s, t))
			if s != t:
				outcome.append(((alpha / beta, s - t), c, t, s))

		# sort likelihood ratio in a descending order, i.e., r1 >= r2 >= ...
		outcome.sort(key=lambda x: x[0][1])
		p_given_lower = 0
		q_given_lower = 0
		# print('Begin Forward...')
		for i in tqdm(range(len(outcome)-1, -1, -1)):
			ratio, cnt, s, t = outcome[i]
			# cnt = int(cnt)
			p = (alpha ** (global_d - s)) * (beta ** s)
			q = (alpha ** (global_d - t)) * (beta ** t)
			q_delta_lower = q * cnt
			p_delta_lower = p * cnt

			if p_given_lower + p_delta_lower < plower_Z:
				p_given_lower += p_delta_lower
				q_given_lower += q_delta_lower
			else:
				q_given_lower += (plower_Z - p_given_lower) / (Decimal(ratio[0]) ** ratio[1])
				break
		q_given_lower /= total_Z

		# sort likelihood ratio in a ascending order
		p_given_upper = 0
		q_given_upper = 0
		# print('Begin Backward...')
		for i in tqdm(range(len(outcome))):
			ratio, cnt, s, t = outcome[i]
			# cnt = int(cnt)
			p = (alpha ** (global_d - s)) * (beta ** s)
			q = (alpha ** (global_d - t)) * (beta ** t)
			q_delta_upper = q * cnt
			p_delta_upper = p * cnt

			if p_given_upper + p_delta_upper < pupper_Z:
				p_given_upper += p_delta_upper
				q_given_upper += q_delta_upper
			else:
				q_given_upper += (pupper_Z - p_given_upper) / (Decimal(ratio[0]) ** ratio[1])
				break
		q_given_upper /= total_Z

		if Decimal(q_given_lower) - Decimal(q_given_upper) < 0:
			return True
		else:
			return False

	p_l, frac_alpha, global_d, v_range, fn, ind = args

	alpha = int(frac_alpha * 100)
	beta = 100 - alpha
	
	def exp_search(v):
		if v == 0:
			return 1
		res_v = test_v(p_l, alpha, beta, global_d, v, fn)
		if res_v:
			return exp_search(int(v // 2))
		else:
			return int(2 * v)
	
	res_start = test_v(p_l, alpha, beta, global_d, v_range[0], fn)
	if res_start:
		return v_range[0]
	res_end = test_v(p_l, alpha, beta, global_d, v_range[1], fn)
	if not res_end:
		return None
	return exp_search(v_range[1])
