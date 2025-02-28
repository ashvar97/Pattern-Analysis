#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

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

def draw_dataset(cdf, ground_truth_x, num_samples, sigma_noise):
    # then: sample, but put some homoscedastic noise in the sampled location
    u = np.sort(np.random.rand(num_samples))
    cdf_ptr = 0
    samples = np.zeros(num_samples)
    for u_pos in range(num_samples):
        while (u[u_pos] > cdf[cdf_ptr]):
            cdf_ptr = cdf_ptr + 1
        samples[u_pos] = ground_truth_x[cdf_ptr]
    # here comes the noise
    samples = samples + np.random.normal(0, sigma_noise, np.shape(samples))
    samples[samples<x_min] = x_min
    samples[samples>x_max] = x_max
    # let's sort them again, this will help our density estimators later
    return np.sort(samples)

def get_parzen_pdf(h, training):
    # let's do the density estimation
    step_width = (x_max-x_min)/n_bins
    parzen_pdf = np.zeros(n_bins)
    sample_positions = ((training-x_min) / step_width).astype(np.uint)
    sample_positions[sample_positions >= n_bins] = n_bins-1
    parzen_pdf[sample_positions] = 1
    parzen_window = np.ones(int(h/step_width))
    parzen_window = parzen_window / np.size(parzen_window) # normalize to 1
    parzen_pdf = np.convolve(parzen_pdf, parzen_window, 'same')
    sum_parzen_pdf = np.sum(parzen_pdf)
    parzen_pdf = parzen_pdf / sum_parzen_pdf
    return (parzen_pdf, step_width)


def evaluate_dataset(testing, parzen_pdf, x_min, x_max, step_width):
    testing = np.sort(testing)
    quality = 0
    for i in range(testing.shape[0]):
        probe_pos = int((testing[i] - x_min) / step_width)
        if ((probe_pos >= 0) and (probe_pos < parzen_pdf.shape[0])):
            likelihood = parzen_pdf[probe_pos]
            if (likelihood > 0):
                quality = quality + np.log(likelihood)
            else:
                # unclean, but avoids -infinity:
                quality = quality + np.log(0.001)
    return quality
    



show_input_data = 1

# let's fix the randomness
np.random.seed(41)

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


# homoscedastic noise
sigma = 0.05

# for representing the density or cdf
n_bins = 500

ground_truth_x = np.linspace(x_min, x_max, n_bins)
ground_truth_y = get_function_values(ground_truth_x)

# sampling process leads to some uncertainty in the exact location of an observation
#
# first: calculate the cdf
cdf = np.zeros(n_bins)
cdf[0] = ground_truth_y[0]
for i in range(1,np.shape(ground_truth_y)[0]):
    cdf[i] = cdf[i-1] + ground_truth_y[i]
# normalization to 1 should theoretically not be critical since the pdf should
# be normalized. However, in practice our normalization does not account for
# discretization; better safe than sorry:
cdf = cdf / cdf[-1]
 

training = draw_dataset(cdf, ground_truth_x, num_samples, sigma) 

# h_range should go from 0.1 to 4.0, but due to the index trickery in
# log_likelihoods[h-h_range[0]] below everything is multiplied here by 10 such
# that these are integers.
# Hence, don't forget to divide h by 10 in the fct call (/10.0)
h_range = np.array(range(1, 40)) 
log_likelihoods = np.zeros(h_range.shape[0])

for h in h_range:
    (parzen_pdf,step_width) = get_parzen_pdf(h/10.0, training)

    testing = draw_dataset(cdf, ground_truth_x, 50, sigma) 
    avg_log_likelihood = evaluate_dataset(testing, parzen_pdf, x_min, x_max, step_width)
    for i in range(19):
        testing = draw_dataset(cdf, ground_truth_x, 50, sigma) 
        tmp_log_likelihood = evaluate_dataset(testing, parzen_pdf, x_min, x_max, step_width)
        avg_log_likelihood = avg_log_likelihood + tmp_log_likelihood
    avg_log_likelihood = avg_log_likelihood / 20
    log_likelihoods[h-h_range[0]] = avg_log_likelihood 
    print(f"h = \t{h}: \t {avg_log_likelihood}")

# let's check for the training error

if (show_input_data != 0):
    fig, ax = plt.subplots(1,1)
    ax.set_xlim(h_range[0]/10, h_range[-1]/10)

    ax.plot( h_range/10, log_likelihoods, 'g', linewidth=3)
    plt.show()


