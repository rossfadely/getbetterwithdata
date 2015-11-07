from matplotlib import use; use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from sklearn.mixture import GMM
from sklearn.neighbors import KernelDensity

def fit_gmm(costs, k=3, min_samps=50):
    """
    Fit return a scikit GMM model.
    """
    if costs.size < min_samps:
        raise Exception('Not enough samps')
    
    gmm = GMM(k)
    return gmm.fit(costs)

def fit_kde(costs, frac_std):
    """
    Fit a KDE to the costs, use a gaussian kernel and a bandwidth that is the
    specified fraction of the std.
    """
    bw = frac_std * np.std(costs)
    kde = KernelDensity(bandwidth=bw)
    return kde.fit(costs)

def select_conditionals(df, features):
    """
    Return the costs associated with the selected features.
    """
    costs = np.array(df['Total Charges'])
    keys = features.keys()
    ind = np.zeros(df.shape[0], np.bool)
    print features
    print ind.shape
    print keys
    #arr = np.array(df[keys[0]])
    print arr.shape
    for i in range(len(keys)):
        arr = np.array(df[keys[i]], type(features[keys[i]]))
        ind = ind & (arr == features[keys[i]])

    print arr[ind].shape
    print costs.shape
    print costs[ind].shape
    assert 0
    return costs[ind]

def plot_gmm(train, test, gmm):
    """
    Plot the train and test data and the empirical estimate.
    """
    rng = [0, 0]
    rng[0] = 0.95 * np.minimum(train.min(), test.min())
    rng[1] = 1.05 * np.maximum(train.max(), train.max())
    assert 0

def plot_gmm(train, test, kde, nsamps, fs=5, filename='./plots/kde.png'):
    """
    Plot the train and test data and the empirical estimate.
    """
    rng = [0, 0]
    rng[0] = 0.95 * np.minimum(train.min(), test.min())
    rng[1] = 1.05 * np.maximum(train.max(), train.max())
    samps = kde.sample(nsamp)
    scores = kde.score(samps)
    
    fig = pl.figure(figsize=(2 * fs, fs))
    pl.subplot(121)
    pl.hist(train, bins=np.min(100, 1. * train.size / 10), normed=True,
            color='k', alpha=0.4)
    pl.plot(samps, scores, 'r', lw=2)
    pl.subplot(121)
    pl.hist(train, bins=np.min(100, 1. * train.size / 10), normed=True,
            color='k', alpha=0.4)
    pl.plot(samps, scores, 'r', lw=2)
    fig.savefig(filename)

def get_rand_features():
    assert 0

if __name__ == '__main__':

    seed = 234
    np.random.seed(seed)

    df = pd.read_csv('./data/sparcs_cleaned.csv')
    keys = list(df.columns.values)
    
    ind = np.random.randint(len(df.shape[0]))
    features = {}
    for k in keys:
        features[k] = df[k][ind]
    features.pop('Unnamed: 0')
    features.pop('Total Charges')



