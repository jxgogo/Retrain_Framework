import scipy
import scipy.linalg as la
import numpy as np
from scipy.io import loadmat


def shuffle_data(data_size, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    indices = np.arange(data_size)
    return np.random.permutation(indices)


def EA(data):
    """data alignment"""
    data_align = []
    length = len(data)
    rf_matrix = np.dot(data[0], np.transpose(data[0]))
    for i in range(1, length):
        rf_matrix += np.dot(data[i], np.transpose(data[i]))
    rf_matrix /= length

    rf = la.inv(la.sqrtm(rf_matrix))
    if rf.dtype == complex:
        rf = rf.astype(np.float64)

    for i in range(length):
        data_align.append(np.dot(rf, data[i]))

    return np.asarray(data_align)


def standard_normalize(x, clip_range=None):
    x = (x - np.mean(x)) / np.std(x)
    if clip_range is not None:
        x = np.clip(x, a_min=clip_range[0], a_max=clip_range[1])
    return x


def random_split(x, y, ratio):
    idx = np.random.permutation(np.arange(len(x)))
    lb_idx = idx[:int(ratio * len(x))]
    while len(np.unique(y[idx])) != len(np.unique(y)):
        idx = np.random.permutation(np.arange(len(x)))
        lb_idx = idx[:int(ratio * len(x))]
    ulb_idx = np.array(sorted(list(set(range(len(x))) - set(lb_idx))))
    return lb_idx, ulb_idx


def BNCILoad(data_path, id, lb_ratio=0.05, align=''):
    label_dict_1 = {
        'left_hand': 0,
        'right_hand': 1,
    }
    label_dict_2 = {
        'left_hand': 0,
        'right_hand': 1,
        'feet': 2,
        'tongue': 3
    }
    label_dict_3 = {
        'right_hand': 0,
        'feet': 1
    }
    if 'BNCI2014-001-2' in data_path: label_dict = label_dict_1
    elif 'BNCI2014-001-4' in data_path: label_dict = label_dict_2
    elif 'BNCI2014-002-2' in data_path: label_dict = label_dict_3
    elif 'BNCI2015-001-2' in data_path: label_dict = label_dict_3
    data = scipy.io.loadmat(data_path + f'A{id + 1}.mat')
    x, y = data['X'], data['y']

    try:
        y = np.array([label_dict[y[j].replace(' ', '')] for j in range(len(y))]).reshape(-1)
    except TypeError or KeyError:
        y = y.reshape(-1)
    lb_idx, ulb_idx = random_split(x, y, ratio=lb_ratio)
    x_lb, y_lb = x[lb_idx], y[lb_idx]
    # use EA on obtained labeled data
    if align == 'EA':
        x_lb = EA(x_lb)
    return x_lb, y_lb


