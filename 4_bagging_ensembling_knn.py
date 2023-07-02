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
    # let's sort them again, this will help our density estimators later
    return np.sort(samples)

def get_knn_pdf(k, training):
    step_width = (x_max-x_min)/n_bins
    knn_pdf = np.zeros(n_bins)
    n_samples = np.size(training)
    last_neighbor = int(0)
    first_neighbor = int(k-1)

    # theoretically, all neighbors could be left of x_min
    # we have to catch this with an initial iteration
    cur_V = _max(np.abs(training[0] - training[last_neighbor]), np.abs(training[0] - training[first_neighbor]))
    next_V = np.Inf
    if (first_neighbor+1 < n_samples):
        next_V = _max(np.abs(training[0] - training[last_neighbor+1]), np.abs(training[0] - training[first_neighbor+1]))
    else:
        next_V = np.Inf
    while (cur_V > next_V):
        last_neighbor = last_neighbor + 1
        first_neighbor = first_neighbor + 1
        cur_V = next_V
        if (first_neighbor+1 < n_samples):
            next_V = _max(np.abs(training[0] - training[last_neighbor+1]), np.abs(training[0] - training[first_neighbor+1]))
        else:
            next_V = np.Inf

    # now calculate the density in the domain of knn_pdf
    for i in range(n_bins):
        # drop last neighbor in favor of next neighbor?
        cur_pos = training[0] + i*step_width
        cur_V   = _max(np.abs(cur_pos - training[last_neighbor]), np.abs(cur_pos - training[first_neighbor]))
        if (first_neighbor+1 < n_samples):
            next_V  = _max(np.abs(cur_pos - training[last_neighbor+1]), np.abs(cur_pos - training[first_neighbor+1]))
            # it could be that closer samples got into range
            while (cur_V > next_V):
                last_neighbor = last_neighbor + 1
                first_neighbor = first_neighbor + 1
                cur_V = next_V
                if (first_neighbor+1 < n_samples):
                    next_V = _max(np.abs(cur_pos - training[last_neighbor+1]), np.abs(cur_pos - training[first_neighbor+1]))
                else:
                    next_V = np.Inf
        knn_pdf[i] = k/((cur_V+0.001) * n_samples)

    return (knn_pdf / np.sum(knn_pdf), step_width)


def evaluate_dataset(testing, knn_pdf, x_min, x_max, step_width):
    testing = np.sort(testing)
    quality = 0
    for i in range(testing.shape[0]):
        probe_pos = int((testing[i] - x_min) / step_width)
        if ((probe_pos >= 0) and (probe_pos < knn_pdf.shape[0])):
            likelihood = knn_pdf[probe_pos]
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
bagging_percentage = 0.5


# homoscedastic noise
sigma = 0.05

# for representing the density or cdf
n_bins = 500

ground_truth_x = np.linspace(x_min, x_max, n_bins)
ground_truth_y = get_function_values(ground_truth_x)

# sampling process leads to some uncertainty in the exact location of an observation
#
# first: calculate the cdf
cdf = np.linspace(x_min, x_max, n_bins)
cdf[0] = ground_truth_y[0]
for i in range(1,np.shape(ground_truth_y)[0]):
    cdf[i] = cdf[i-1] + ground_truth_y[i]
# normalization to 1 should theoretically not be critical since the pdf should
# be normalized. However, in practice our normalization does not account for
# discretization; better safe than sorry:
cdf = cdf / cdf[-1]
 
base_dataset = draw_dataset(cdf, ground_truth_x, int(num_samples/bagging_percentage), sigma) 

k_range = np.array(range(3,40))
log_likelihoods = np.zeros(k_range.shape[0])

for k in k_range:

    # i.d. ensembling means that we draw only a bagged subset from our data
    rng = np.random.default_rng()
    training = rng.permutation(base_dataset)[0:num_samples]
    (avg_knn_pdf, step_width) = get_knn_pdf(k, training)
    for i in range(19):
        training = rng.permutation(base_dataset)[0:num_samples]
        (tmp_knn_pdf, step_width) = get_knn_pdf(k, training)
        avg_knn_pdf = avg_knn_pdf + tmp_knn_pdf
    knn_pdf = avg_knn_pdf / 20    

    testing = draw_dataset(cdf, ground_truth_x, 50, sigma) 
    avg_log_likelihood = evaluate_dataset(testing, knn_pdf, x_min, x_max, step_width)
    for i in range(19):
        testing = draw_dataset(cdf, ground_truth_x, 50, sigma) 
        tmp_log_likelihood = evaluate_dataset(testing, knn_pdf, x_min, x_max, step_width)
        avg_log_likelihood = avg_log_likelihood + tmp_log_likelihood
    avg_log_likelihood = avg_log_likelihood / 20
    log_likelihoods[k-k_range[0]] = avg_log_likelihood 
    print(f"k = \t{k}: \t {avg_log_likelihood}")

# let's check for the training error
if (show_input_data != 0):
    fig, ax = plt.subplots(1,1)
    ax.set_xlim(k_range[0], k_range[-1])

    ax.plot( k_range, log_likelihoods, 'g', linewidth=3)
    plt.show()

