#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

show_input_data = 1

# let's fix the randomness
np.random.seed(45)

# parameters for our synthetic ground truth function
num_dimensions = 3
ground_truth_coefficients = np.zeros((num_dimensions, 1))

# fuer num_dimensions = 3:
ground_truth_coefficients[0] = 3 
ground_truth_coefficients[1] = 0.3
ground_truth_coefficients[2] = 0.1

# get random positions between 0 and 3
x_min = 0
x_max = 10

num_samples = 200

# input_positions must be a column vector of x-positions where the function is
# evaluated
def get_function_values(input_positions):
    output = np.sin(input_positions*ground_truth_coefficients[0]) * input_positions * ground_truth_coefficients[1] + ground_truth_coefficients[2]*input_positions + 2
#    return output
    vec_sum = np.sum(output)
    return output/vec_sum

# this surely exists in the python API but I can't find it right now
def _max(a, b):
    if (a > b):
        return a
    return b


# random sample positions (influence with the seed on top ^)
x_pos = (np.random.rand(num_samples, 1) * (x_max-x_min)) - x_min

# homoscedastic noise
sigma = 0.05

# for representing the density or cdf
n_bins = 500

ground_truth_x = np.linspace(x_min, x_max, n_bins)
ground_truth_y = get_function_values(ground_truth_x)
sum_gty = np.sum(ground_truth_y)
ground_truth_y = ground_truth_y / sum_gty

# sampling process leads to some uncertainty in the exact location of an observation
#
# first: calculate the cdf
cdf = np.zeros(n_bins)
cdf[0] = ground_truth_y[0]
for i in range(1,np.shape(ground_truth_y)[0]):
    cdf[i] = cdf[i-1] + ground_truth_y[i]
# normalization to 1 is not critical since the pdf is normalized
# however, normalization is somewhat tricky, because discretization step with
# also comes into play
cdf = cdf / cdf[-1]
 
# then: sample, but put some homoscedastic noise in the sampled location
u = np.sort(np.random.rand(num_samples))
cdf_ptr = 0
samples = np.zeros(num_samples)
for u_pos in range(num_samples):
    while (u[u_pos] > cdf[cdf_ptr]):
        cdf_ptr = cdf_ptr + 1
    samples[u_pos] = ground_truth_x[cdf_ptr]
# here comes the noise
samples = samples + np.random.normal(0, sigma, np.shape(samples))
# let's sort them again, this will help our density estimators later
samples = np.sort(samples)

# let's do the density estimation
# kNN:
k = 7
step_width = (x_max-x_min)/n_bins
knn_pdf = np.zeros(n_bins)
n_samples = np.size(samples)
last_neighbor = int(0)
first_neighbor = int(k-1)

# theoretically, all neighbors could be left of x_min
# we have to catch this with an initial iteration
cur_V = _max(np.abs(samples[0] - samples[last_neighbor]), np.abs(samples[0] - samples[first_neighbor]))
next_V = np.Inf
if (first_neighbor+1 < n_samples):
    next_V = _max(np.abs(samples[0] - samples[last_neighbor+1]), np.abs(samples[0] - samples[first_neighbor+1]))
else:
    next_V = np.Inf
while (cur_V > next_V):
    last_neighbor = last_neighbor + 1
    first_neighbor = first_neighbor + 1
    cur_V = next_V
    if (first_neighbor+1 < n_samples):
        next_V = _max(np.abs(samples[0] - samples[last_neighbor+1]), np.abs(samples[0] - samples[first_neighbor+1]))
    else:
        next_V = np.Inf

# now calculate the density in the domain of knn_pdf
for i in range(n_bins):
    # drop last neighbor in favor of next neighbor?
    cur_pos = samples[0] + i*step_width
    cur_V   = _max(np.abs(cur_pos - samples[last_neighbor]), np.abs(cur_pos - samples[first_neighbor]))
    if (first_neighbor+1 < n_samples):
        next_V  = _max(np.abs(cur_pos - samples[last_neighbor+1]), np.abs(cur_pos - samples[first_neighbor+1]))
        # it could be that closer samples got into range
        while (cur_V > next_V):
            last_neighbor = last_neighbor + 1
            first_neighbor = first_neighbor + 1
            cur_V = next_V
            if (first_neighbor+1 < n_samples):
                next_V = _max(np.abs(cur_pos - samples[last_neighbor+1]), np.abs(cur_pos - samples[first_neighbor+1]))
            else:
                next_V = np.Inf
    knn_pdf[i] = k/((cur_V+0.001) * n_samples)

knn_pdf = knn_pdf / np.sum(knn_pdf)



if (show_input_data != 0):
    fig, ax = plt.subplots(1,1)
    ax.scatter(samples.reshape(num_samples,), np.ones(num_samples,).reshape(num_samples,) * 0.005) 
    ax.set_xlim(x_min, x_max)

    ax.plot( ground_truth_x, ground_truth_y, 'g', linewidth=3)
    ax.plot( ground_truth_x, knn_pdf, 'r', linewidth=3)
    plt.show()


