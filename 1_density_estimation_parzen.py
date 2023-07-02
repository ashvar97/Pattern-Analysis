#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

# input_positions must be a column vector of x-positions where the function is
# evaluated
def get_function_values(input_positions):
    output = np.sin(input_positions*ground_truth_coefficients[0]) * input_positions * ground_truth_coefficients[1] + ground_truth_coefficients[2]*input_positions + 2
    sum_entries = np.sum(output)
    return output/sum_entries



show_input_data = 1

# let's fix the randomness
np.random.seed(43)

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

num_samples = 20

# initialize first column with value 1, second column with randomly drawn x positions
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
# for this we probably require manual sampling
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
# plus homoscedastic Gaussian noise
samples = samples + np.random.normal(0, sigma, np.shape(samples))
samples[samples<x_min] = x_min
samples[samples>x_max] = x_max
# let's sort them again, this will help our density estimators later
samples = np.sort(samples)


# let's do the density estimation
# parzen windows:
h = 0.1
# discretization step width
step_width = (x_max-x_min)/n_bins
parzen_pdf = np.zeros(n_bins)
sample_positions = ((samples-x_min) / step_width).astype(np.uint)
sample_positions[sample_positions >= n_bins] = n_bins-1
parzen_pdf[sample_positions] = 1
parzen_window = np.ones(int(h/step_width))
parzen_window = parzen_window / np.sum(parzen_window) # normalize to 1
parzen_pdf = np.convolve(parzen_pdf, parzen_window, 'same')
sum_parzen_pdf = np.sum(parzen_pdf)
parzen_pdf = parzen_pdf / sum_parzen_pdf


if (show_input_data != 0):
    fig, ax = plt.subplots(1,1)
    ax.scatter(samples.reshape(num_samples,), np.ones(num_samples,).reshape(num_samples,) * 1.1 * np.max(parzen_pdf))
    ax.set_xlim(x_min, x_max)

    # green ground truth curve, normalized to 1
    ax.plot( ground_truth_x, ground_truth_y, 'g', linewidth=3)
    # kernel window, just for reference
    ax.plot( ground_truth_x, np.pad(parzen_window, (40, n_bins-40-np.size(parzen_window)), 'constant', constant_values=(0, 0)) * ( 0.3*np.max(parzen_pdf)/ np.max(parzen_window)), 'y', linewidth=3)
    ax.plot( ground_truth_x, parzen_pdf, 'r', linewidth=3)
    plt.show()


