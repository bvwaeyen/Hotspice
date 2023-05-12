# Created 27/02/2023 by Ian Lateur
# Code is made by samueljackson92 and copied from https://github.com/samueljackson92/coranking

# The ideas are based on the paper from Lee, John A., and Michel Verleysen.
# "Quality assessment of dimensionality reduction: Rank-based criteria." Neurocomputing 72.7 (2009): 1431-1443.

# Why copy instead of installing?
# 1) Installing didn't work instantly
# 2) It needs to be modified for hotspice


import numpy as np
from scipy.spatial import distance


def coranking_matrix(original_data, transformed_data, original_metric=None, transformed_metric=None):
    """Generate a co-ranking matrix from two sets of datapoints if metrics are specified or two sets of distances
    if the metrics are None.
    @param original_data: Either an n by m array of n original datapoints in an m-dimensional space if metrics are
        specified or a 1D array of n(n-1)/2 distances if metrics are None.
    @param transformed_data: See original_data, but for different dimensional space.
    @param original_metric: distance metric for calculating distances in the original space if not None.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    @param transformed_metric: distance metric for calculating distances in the transformed space if not None.
    :returns: the co-ranking matrix of the two data sets.
    """

    if original_metric is not None:  # calculate distances from data, else already have distances
        if transformed_metric is None: transformed_metric = original_metric
        original_data = distance.pdist(original_data, metric=original_metric)
        transformed_data = distance.pdist(transformed_data, metric=transformed_metric)

    original_distance = distance.squareform(original_data)
    transformed_distance = distance.squareform(transformed_data)

    original_ranking = original_distance.argsort(axis=1).argsort(axis=1)
    transformed_ranking = transformed_distance.argsort(axis=1).argsort(axis=1)

    n, _ = original_distance.shape  # number of points, number of points
    Q, xedges, yedges = np.histogram2d(original_ranking.flatten(), transformed_ranking.flatten(), bins=n)

    Q = Q[1:, 1:]  # remove rankings which correspond to themselves
    return Q

# ----------------------------------------------------------------------------------------------------------------------

# TODO: max_k does not need -1, but it behaves weirdly without it?

# Metrics

def trustworthiness(Q, K):
    """ The trustworthiness measure complements continuity and is a measure of
    the number of hard intrusions.
    Args:
        Q: the co-ranking matrix to calculate continuity from
        k (int): the number of neighbours to use.
    Returns:
        The trustworthiness metric for the given K
    """
    n = Q.shape[0]
    summation = 0.0

    norm_weight = _tc_normalisation_weight(K, n + 1)
    w = 2.0 / norm_weight

    for k in range(K, n):
        for l in range(K):
            summation += w * (k + 1 - K) * Q[k, l]

    return 1.0 - summation

def trustworthiness_range(Q, min_k=1, max_k=None):
    """Compute the trustwortiness metric over a range of K values.

    :param Q: coranking matrix
    :param min_k: the lowest K value to compute. Default 1.
    :param max_k: the highest K value to compute. If None the range of values
        will be computer from min_k to n-1

    :returns: array of size min_k - max_k with the corresponding
        trustworthiness values.
    """
    if not isinstance(Q, np.int64):
        Q = Q.astype(np.int64)

    if max_k is None:
        max_k = Q.shape[0]-1

    result = [trustworthiness(Q, x) for x in range(min_k, max_k)]
    return np.array(result)

# ----------------------------------------------------------------------------------------------------------------------

def continuity(Q, K):
    """ The continutiy measure complements trustworthiness and is a measure of
    the number of hard extrusions.

    Args:
        Q: the co-ranking matrix to calculate continuity from
        k (int): the number of neighbours to use.

    Returns:
        The continuity metric for the given K
    """
    n = Q.shape[0]
    summation = 0.0

    norm_weight = _tc_normalisation_weight(K, n + 1)
    w = 2.0 / norm_weight

    for k in range(K):
        for l in range(K, n):
            summation += w * (l + 1 - K) * Q[k, l]

    return 1.0 - summation

def continuity_range(Q, min_k=1, max_k=None):
    """Compute the continuity metric over a range of K values.

    :param Q: coranking matrix
    :param min_k: the lowest K value to compute. Default 1.
    :param max_k: the highest K value to compute. If None the range of values
        will be computer from min_k to n-1

    :returns: array of size min_k - max_k with the corresponding continuity
        values.
    """
    if not isinstance(Q, np.int64):
        Q = Q.astype(np.int64)

    if max_k is None:
        max_k = Q.shape[0]-1

    result = [continuity(Q, x) for x in range(min_k, max_k)]
    return np.array(result)

# ----------------------------------------------------------------------------------------------------------------------

def LCMC(Q, K):  # FIXME: see bugreport https://github.com/samueljackson92/coranking/issues/10
    """ The local continuity meta-criteria measures the number of mild
    intrusions and extrusions. This can be thought of as a measure of the
    number of true postives.

    Args:
        Q: the co-ranking matrix to calculate continuity from
        k (int): the number of neighbours to use.

    Returns:
        The LCMC metric for the given K
    """
    n = Q.shape[0]
    summation = 0.0

    for k in range(K):
        for l in range(K):
            summation += Q[k, l]

    return (K / (1. - n)) + (1. / (n * K)) * summation

def LCMC_range(Q, min_k=1, max_k=None):
    """Compute the local continuity meta-criteria (LCMC) metric over a range of
    K values.

    :param Q: coranking matrix
    :param min_k: the lowest K value to compute. Default 1.
    :param max_k: the highest K value to compute. If None the range of values
        will be computer from min_k to n-1

    :returns: array of size min_k - max_k with the corresponding LCMC values.
    """
    if not isinstance(Q, np.int64):
        Q = Q.astype(np.int64)

    if max_k is None:
        max_k = Q.shape[0]-1

    result = [LCMC(Q, x) for x in range(min_k, max_k)]
    return np.array(result)

# ----------------------------------------------------------------------------------------------------------------------

def _tc_normalisation_weight(K, n):
    """ Compute the normalisation weight for the trustworthiness and continuity
    measures.

    Args:
        K (int): size of the neighbourhood
        n (int): total size of the matrix

    Returns:
        Normalisation weight for trustworthiness and continuity metrics
    """
    if K < (n / 2):
        return n * K * (2 * n - 3 * K - 1)
    elif K >= (n / 2):
        return n * (n - K) * (n - K)

def _check_square_matrix(M):
    if M.shape[0] != M.shape[1]:
        msg = "Expected square matrix, but matrix had dimensions (%d, %d)" % M.shape
        raise RuntimeError(msg)