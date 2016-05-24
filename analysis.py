from time import time
from datetime import datetime as dt
import pickle

import numpy as np
from scipy.stats import multivariate_normal as mnorm
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import cm


def calc_logpdfs(data, mus, sigmas):
    logpdfs = np.zeros((len(data), len(mus)))
    for k in range(len(mus)):
        logpdfs[:,k] = mnorm.logpdf(data, mus[k], sigmas[k])
    return logpdfs


def calc_probs(logpdfs, pi):
    weighted_pdfs = (pi * np.exp(logpdfs))
    tot_pdfs = np.sum(weighted_pdfs, axis=1)
    probs = weighted_pdfs / tot_pdfs[:, np.newaxis]
    return probs


def gmm(data, init_mu, init_sigma, init_pi, max_iter=100, diff_tol=1e-3):
    curr_mu = init_mu.copy()
    curr_sigma = init_sigma.copy()
    curr_pi = init_pi

    num_iter = 0
    diff = 10
    logliks = []
    while num_iter < max_iter and diff > diff_tol:
        # E-step
        logpdfs = calc_logpdfs(data, curr_mu, curr_sigma)
        probs = calc_probs(logpdfs, curr_pi)
        logliks.append(np.sum(np.log(np.sum(curr_pi*np.exp(logpdfs), axis=1))))
        # logliks.append(np.sum(probs * logpdfs))

        # M-step
        for k in range(len(init_mu)):
            curr_mu[k] = np.average(data, axis=0, weights=probs[:,k])
            # curr_sigma[k] = np.cov(data, rowvar=0, aweights=probs[:,k], ddof=0)

            dm = data - curr_mu[k]
            curr_sigma[k] = np.dot(probs[:,k]*dm.T, dm) / np.sum(probs[:,k])
        curr_pi = np.mean(probs, axis=0)

        if num_iter >= 2:
            diff = np.abs(logliks[-1] - logliks[-2])

        print(num_iter, logliks[-1])
        num_iter += 1

    logpdfs = calc_logpdfs(data, curr_mu, curr_sigma)
    probs = calc_probs(logpdfs, curr_pi)
    logliks.append(np.sum(np.log(np.sum(curr_pi*np.exp(logpdfs), axis=1))))

    return (curr_mu, curr_sigma, curr_pi), probs, logliks



def gmm_cat(data_raw, cat_raw, init_mu, init_sigma, init_pi=None,
        max_iter=100, diff_tol=1e-3):
    num_cat = len(np.unique(cat_raw))
    num_groups = len(init_mu)

    sort_ind = np.argsort(cat_raw)
    cat = cat_raw[sort_ind]
    data = data_raw[sort_ind]

    cat_bounds = []
    for val in np.unique(cat):
        where = np.where(cat == val)
        cat_bounds.append((np.min(where), np.max(where)+1))

    curr_mu = init_mu.copy()
    curr_sigma = init_sigma.copy()
    if init_pi:
        curr_pi = init_pi
    else:
        curr_pi = np.ones((num_cat, num_groups))/num_groups

    num_iter = 0
    diff = 10
    logliks = []
    while num_iter < max_iter and diff > diff_tol:
        # E-step
        logpdfs = calc_logpdfs(data, curr_mu, curr_sigma)

        probs = np.empty(logpdfs.shape)
        for g, (l, u) in enumerate(cat_bounds):
            probs[l:u] = calc_probs(logpdfs[l:u], curr_pi[g])
        logliks.append(np.sum(np.log(
            np.sum(curr_pi[cat]*np.exp(logpdfs), axis=1))))

        # M-step
        for k in range(num_groups):
            curr_mu[k] = np.average(data, axis=0, weights=probs[:,k])

            dm = data - curr_mu[k]
            curr_sigma[k] = np.dot(probs[:,k]*dm.T, dm) / np.sum(probs[:,k])
        for g, (l, u) in enumerate(cat_bounds):
            curr_pi[g] = np.mean(probs[l:u], axis=0)

        if num_iter >= 2:
            diff = np.abs(logliks[-1] - logliks[-2])

        print(num_iter, logliks[-1])
        num_iter += 1

    return (curr_mu, curr_sigma, curr_pi), probs, logliks


sdata = pd.read_pickle('./data/data_week.p')
sd = np.load('./data/data_week_c.npy')
hour = np.load('./data/data_week_hour.npy')

hour_cat = np.zeros(len(hour), dtype=int)           # Early morning
hour_cat[np.logical_and( 6 <= hour, hour <  9)] = 1 # Morning rush
hour_cat[np.logical_and( 9 <= hour, hour < 16)] = 2 # Work day
hour_cat[np.logical_and(16 <= hour, hour < 19)] = 3 # Evening rush
hour_cat[np.logical_and(19 <= hour, hour < 22)] = 4 # Evening
hour_cat[np.logical_or( 22 <= hour, hour <  2)] = 5 # Night

# np.random.seed(2634)
# inds = np.random.choice(sd.shape[0], 100000, replace=False)
# sd = sd[inds]
# hour = hour[inds]
# hour_cat = hour_cat[inds]


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
morning_rush = np.array(np.logical_and(7 <= hour, hour < 9))
ax.scatter(sd[morning_rush, 0], sd[morning_rush, 1], alpha=0.2, lw=0.6, s=10)
ax.set_xlim(-74.1, -73.7)
ax.set_ylim(40.5, 40.9)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
evening_rush = np.array(np.logical_and(16 <= hour, hour < 18))
ax.scatter(sd[evening_rush, 0], sd[evening_rush, 1], alpha=0.2, lw=0.6, s=10)
ax.set_xlim(-74.1, -73.7)
ax.set_ylim(40.5, 40.9)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
evening_rush = np.array(np.logical_and(22 <= hour, hour < 24))
ax.scatter(sd[evening_rush, 0], sd[evening_rush, 1], alpha=0.2, lw=0.6, s=10)
ax.set_xlim(-74.1, -73.7)
ax.set_ylim(40.5, 40.9)

plt.plot(np.unique(hour, return_counts=True)[0]+0.5,
         np.unique(hour, return_counts=True)[1], color='black')
for x in [2, 6, 9, 16, 19, 22]:
    plt.axvline(x=x)



k = 16

# K-Means
km = KMeans(k, n_jobs=-2)
start = time()
km.fit(sd)
print(time() - start)

pred = km.predict(sd)
fig = plt.figure()
ax = fig.add_subplot(111)
for g, c in zip(range(k), cm.Paired(np.linspace(0, 1, k))):
    ax.scatter(sd[pred==g, 0], sd[pred==g, 1], c=c)


# # Run GMM
# init_sigma = [np.cov(sd, rowvar=0) for i in range(k)]
# init_pi = np.array([1 for i in range(k)])/k
# init_mu = km.cluster_centers_

# np.random.seed(2046)
# start = time()
# res = gmm(sd, init_mu, init_sigma, init_pi, diff_tol=1e-2, max_iter=1000)
# print(time() - start)

# iters = list(range(len(res[2])))
# logliks = res[2]



# Run GMM with categories
init_sigma = [np.cov(sd, rowvar=0) for i in range(k)]
init_mu = km.cluster_centers_

np.random.seed(2046)
start = time()
res_cat = gmm_cat(sd, hour_cat, init_mu, init_sigma,
                  diff_tol=1e-2, max_iter=2000)
print(time() - start)


with open('./res_cat16.p', 'wb') as f_:
    pickle.dump(res_cat, f_)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(sd[:,0], sd[:,1])
ax.scatter(res_cat[0][0][:,0], res_cat[0][0][:,1], color='red')

