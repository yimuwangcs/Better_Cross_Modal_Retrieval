import numpy as np
import torch
from scipy.stats import skew
import numpy as np
from copy import deepcopy

######### Credit to QBNorm (https://github.com/ioanacroi/qb-norm/blob/master/dynamic_inverted_softmax.py)

# Returns list of retrieved top k videos based on the sims matrix
def get_retrieved_videos(sims, k):
    argm = np.argsort(-sims, axis=1)
    topk = argm[:,:k].reshape(-1)
    retrieved_videos = np.unique(topk)
    return retrieved_videos

# Returns list of indices to normalize from sims based on videos
def get_index_to_normalize(sims, videos):
    argm = np.argsort(-sims, axis=1)[:,0]
    result = np.array(list(map(lambda x: x in videos, argm)))
    result = np.nonzero(result)
    return result

def qb_norm(train_test, test_test, k=1, beta=20):
    retrieved_videos = get_retrieved_videos(train_test, k)
    test_test_normalized = test_test
    train_test = np.exp(train_test*beta)
    test_test = np.exp(test_test*beta)

    normalizing_sum = np.sum(train_test, axis=0)
    index_for_normalizing = get_index_to_normalize(test_test, retrieved_videos)
    test_test_normalized[index_for_normalizing, :] = \
        np.divide(test_test[index_for_normalizing, :], normalizing_sum)
    return test_test_normalized

######### IS and DB_Norm

def get_skewness(sim_matrix, topk=10):
    if type(sim_matrix) == torch.Tensor:
        retrieve_index = (-sim_matrix).argsort(axis=1)[:, :topk].reshape(-1).numpy().tolist()
    else:
        retrieve_index = (-sim_matrix).argsort(axis=1)[:, :topk].flatten().tolist()
    count = Counter(retrieve_index)
    l = [count[k] for k in count.keys()]
    return skew(l)

def IS(train_test, test_test, beta=20, k=1):
    test_test_normalized = test_test
    train_test = np.exp(train_test*beta)
    test_test = np.exp(test_test*beta)

    normalizing_sum = np.sum(train_test, axis=0)
    test_test_normalized = test_test / normalizing_sum
    return test_test_normalized

def db_norm(
    train_cap_test_vis,
    train_vis_test_vis,
    test_cap_test_vis,
    k=1,
    beta1=1 / 1.99,
    beta2=3,
    dynamic_normalized=False
):
    func = qb_norm if dynamic_normalized else IS

    test_test_query_normalized = func(train_cap_test_vis, train_vis_test_vis, k=k, beta=beta1)
    test_test_gallery_normalized = func(train_vis_test_vis, train_vis_test_vis, k=k, beta=beta2)

    sim_matrix_normalized = test_test_query_normalized * test_test_gallery_normalized

    return sim_matrix_normalized
