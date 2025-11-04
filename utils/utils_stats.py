import numpy as np

import scipy.stats as stats

import sys

from scipy.stats import bootstrap

import matplotlib.pyplot as plt


NB_RESAMPLES = 100000


def mean_confidence_interval(a, 
                             confidence_normality_test = 0.95,
                             confidence_level = 0.95,
                             force = None,
                             random_state = np.random):
    # Check if the test does not reject normality of data
    res_norm = stats.normaltest(a)
    if (force is None and res_norm.pvalue >= 1 - confidence_normality_test) or force == "normal":
        # If the test 'does not reject normality' of data, 
        # calculate confidence interval using student law
        # force == 'normal' correspond to the case when we want to force the use of normal distribution
        ci = stats.t.interval(confidence = confidence_level, 
                            df = len(a)-1, 
                            loc = np.mean(a), 
                            scale = np.std(a, ddof=1) / np.sqrt(len(a)))
    elif (force is None and res_norm.pvalue < 1 - confidence_normality_test) or force == "bootstrap":
        # If the test 'does reject normality' of data 
        # calculate confidence interval using a bootrapped estimate
        # force == 'bootstrap' correspond to the case when we want to force the use of bootstrapping
        res = bootstrap((a,), 
                        np.mean,
                        n_resamples = NB_RESAMPLES, 
                        confidence_level = confidence_level,
                        random_state = random_state)
        ci = (res.confidence_interval.low, res.confidence_interval.high)
    else:
        print("Value of force unrocognized.")
        sys.exit()
    return ci, res_norm.pvalue


size_a = -1
def diff_mean(sample):
    mean1 = np.mean(sample[:size_a])
    mean2 = np.mean(sample[size_a+1:])
    return mean1 - mean2


def test_twoind_sample(a, b,
                       confidence_normality_test = 0.95,
                       confidence_level = 0.95,
                       alternative_hypothesis = "two-sided",
                       force = None,
                       random_state = np.random):
    # Check if the test does not reject normality of data
    res_norm = stats.normaltest(a)
    if (force is None and res_norm.pvalue >= 1 - confidence_normality_test) or force == "normal":
        # force == 'normal' correspond to the case when we want to force the use of normal distribution
        # Do a t-test for the hypothesis
        res = stats.ttest_ind(a, b,
                              alternative = alternative_hypothesis,
                              random_state = random_state,
                              equal_var = False)
        # Construct the return dictionary
        dict_return = {"decision":"reject" if res.pvalue < 1 - confidence_level else "not reject",
                       "pvalue":res.pvalue}
        return dict_return
    elif (force is None and res_norm.pvalue < 1 - confidence_normality_test) or force == "bootstrap":
        # force == 'bootstrap' correspond to the case when we want to force the use of bootstrapping
        # Calculate the obseved mean difference
        obs_diff_mean = np.mean(a) - np.mean(b)
        #print(obs_diff_mean)
        # Calculate the bootstrap distribution
        global size_a
        size_a = len(a)
        res = bootstrap((a+b,), 
                        diff_mean, 
                        method = 'percentile', # here method is 'percentile' because 'BCa' bugs with 'diff_mean' (maybe sizes issues), so easy fix lol
                        n_resamples = NB_RESAMPLES,
                        random_state = random_state)
        # Calculate the pvalue
        nb_resamples = len(res.bootstrap_distribution)
        if alternative_hypothesis == "two-sided":
            pvalue = sum(int(resamp_diff_mean >= abs(obs_diff_mean) or\
                             resamp_diff_mean <= -abs(obs_diff_mean)) 
                                    for resamp_diff_mean in res.bootstrap_distribution)/nb_resamples
        elif alternative_hypothesis == "less":
            pvalue = sum(int(resamp_diff_mean <= obs_diff_mean) 
                                    for resamp_diff_mean in res.bootstrap_distribution)/nb_resamples
        elif alternative_hypothesis == "greater":
            pvalue = sum(int(resamp_diff_mean >= obs_diff_mean) 
                                    for resamp_diff_mean in res.bootstrap_distribution)/nb_resamples
        else:
            print("Hypothesis type unrecognized : ", alternative_hypothesis)
            sys.exit()
        # Construct the return dictionary
        dict_return = {"decision":"reject" if pvalue < 1 - confidence_level else "not reject",
                       "pvalue":pvalue}
        return dict_return
    else:
        print("Value of 'force' is unrecognized.")
        sys.exit()