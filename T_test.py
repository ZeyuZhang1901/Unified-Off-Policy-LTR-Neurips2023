import scipy
import numpy as np


def t_test(data_tuple1, data_tuple2, alpha=0.05):
    """Perform t-test on two data tuples.

    Args:
        data_tuple1: tuple, A tuple of data.
        data_tuple2: tuple, A tuple of data.
        alpha: The significance level.

    Returns:
        A tuple of (t, p), where t is the t-statistics and p is the p-value.
    """
    t, p = scipy.stats.ttest_ind(data_tuple1, data_tuple2, equal_var=False)
    return t, p
