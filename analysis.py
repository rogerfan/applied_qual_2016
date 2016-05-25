from time import time
from datetime import datetime as dt
import pickle

import numpy as np
from scipy.stats import multivariate_normal as mnorm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
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

        # M-step
        for k in range(len(init_mu)):
            curr_mu[k] = np.average(data, axis=0, weights=probs[:,k])

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
    num_dim = data_raw.shape[1]

    sort_ind = np.argsort(cat_raw)
    cat = cat_raw[sort_ind]
    data = data_raw[sort_ind]

    cat_bounds = []
    for val in np.unique(cat):
        where = np.where(cat == val)
        cat_bounds.append((np.min(where), np.max(where)+1))

    curr_mu = init_mu.copy()
    curr_sigma = init_sigma.copy()
    if init_pi is not None:
        curr_pi = init_pi
    else:
        curr_pi = np.ones((num_cat, num_groups))/num_groups

    num_iter = 0
    diff = 10
    logliks = []
    try:
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
                denom = np.sum(probs[:,k]) + 10*np.finfo(float).eps
                temp_sigma = np.dot(probs[:,k]*dm.T, dm) / denom
                curr_sigma[k] = temp_sigma + 1e-8*np.eye(num_dim)
            for g, (l, u) in enumerate(cat_bounds):
                curr_pi[g] = np.mean(probs[l:u], axis=0)

            if num_iter >= 2:
                diff = np.abs(logliks[-1] - logliks[-2])

            print(num_iter, logliks[-1])
            num_iter += 1
    except KeyboardInterrupt:
        pass

    return (curr_mu, curr_sigma, curr_pi), probs, logliks


# Handle data
sdata = pd.read_pickle('./data/data_week.p')
sd = np.load('./data/data_week_c.npy')
hour = np.load('./data/data_week_hour.npy')

hour_cat = np.zeros(len(hour), dtype=int)           # Early morning
hour_cat[np.logical_and( 7 <= hour, hour <  9)] = 1 # Morning rush
hour_cat[np.logical_and( 9 <= hour, hour < 16)] = 2 # Work day
hour_cat[np.logical_and(16 <= hour, hour < 18)] = 3 # Evening rush
hour_cat[np.logical_and(18 <= hour, hour < 22)] = 4 # Evening
hour_cat[np.logical_or( 22 <= hour, hour <  2)] = 5 # Night

# Raw data plots
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
morning_rush = np.array(np.logical_and(7 <= hour, hour < 9))
ax.scatter(sd[morning_rush, 0], sd[morning_rush, 1], alpha=0.08, lw=0.1, s=10)
ax.set_xlim(-74.05, -73.75)
ax.set_ylim(40.55, 40.9)
fig.savefig('./include/diff_morningrush.png', bbox_inches='tight', dpi=150)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
night = np.array(np.logical_and(22 <= hour, hour < 24))
ax.scatter(sd[night, 0], sd[night, 1], alpha=0.08, lw=0.1, s=10)
ax.set_xlim(-74.05, -73.75)
ax.set_ylim(40.55, 40.9)
fig.savefig('./include/diff_night.png', bbox_inches='tight', dpi=150)


time = np.array(
    sdata['pickup_time'].dt.hour + (sdata['pickup_time'].dt.minute/60))

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(1, 1, 1)
n, bins, patches = ax.hist(
    time, 50, normed=1, facecolor='0.5', histtype='stepfilled')
bounds = [0, 2, 7, 9, 16, 18, 22, 24]
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
          '#ff7f00', '#ffff33', '#e41a1c']
for i, c in enumerate(colors):
    ax.plot(bounds[i:(i+2)], [0, 0], color=c,
            linestyle='-', linewidth=15, solid_capstyle='butt')
for x in [2, 7, 9, 16, 18, 22]:
    ax.axvline(x, color='blue')
ax.set_xlim(0, 24)
ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
fig.savefig('./include/time.pdf', bbox_inches='tight')


# Run GMM
k = 30

np.random.seed(2046)
gmm_mod = GMM(n_components=k, covariance_type='full', min_covar=1e-8)
start = time()
gmm_mod.fit(sd)
print(time() - start)

with open('./gmm_mod30.p', 'wb') as f_:
    pickle.dump(gmm_mod, f_)
# with open('./gmm_mod30.p', 'rb') as f_:
#     gmm_mod = pickle.load(f_)


# Run GMM with categories
init_mu = gmm_mod.means_
init_sigma = gmm_mod.covars_
init_pi = np.tile(gmm_mod.weights_, (6, 1))

np.random.seed(2046)
start = time()
res_cat = gmm_cat(sd, hour_cat, init_mu, init_sigma, init_pi,
                  diff_tol=1e-2, max_iter=1500)
print(time() - start)

with open('./res_cat30.p', 'wb') as f_:
    pickle.dump((res_cat[0],), f_)
# with open('./res_cat30.p', 'rb') as f_:
#     res_cat = pickle.load(f_)


# Plots
big = [3, 15, 16]
rest = [i for i in range(30) if i not in big]

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(sd[:,0], sd[:,1], alpha=0.08, lw=0.1, s=10, color='0.8')
ax.scatter(gmm_mod.means_[rest,0], gmm_mod.means_[rest,1],
           color='black', s=25, lw=1)

x = np.linspace(-74.05, -73.75)
y = np.linspace(40.55, 40.9)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = gmm_mod.score_samples(XX)[0]
Z = Z.reshape(X.shape)
ax.contour(X, Y, Z)

ax.scatter(gmm_mod.means_[big,0], gmm_mod.means_[big,1],
           color='red', s=25, lw=1)
ax.set_xlim(-74.05, -73.75)
ax.set_ylim(40.55, 40.9)
fig.savefig('./include/gmm_res.png', bbox_inches='tight', dpi=200)


inter = [0, 1, 3, 16, 26]
airport = [2, 4]
outer = [7, 13, 21]
interest = inter + airport + outer
rest = [i for i in range(30) if i not in interest]

fig = plt.figure(figsize=(8, 7.2))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(sd[:,0], sd[:,1], alpha=0.08, lw=0.1, s=10, color='0.8')
ax.scatter(res_cat[0][0][rest,0], res_cat[0][0][rest,1],
            color='black', s=25, lw=1)
ax.scatter(res_cat[0][0][interest,0], res_cat[0][0][interest,1],
            color='blue', s=25, lw=1)
for i, j in enumerate(interest):
    ax.annotate(str(i), xy=res_cat[0][0][j], color='blue',
                xytext=res_cat[0][0][j]+np.array([0.002, 0.002]))
ax.set_xlim(-74.05, -73.75)
ax.set_ylim(40.55, 40.9)
fig.savefig('./include/gmm_cat_res.png', bbox_inches='tight', dpi=200)


gmm_morn = GMM(n_components=k, covariance_type='full', min_covar=1e-8)
gmm_morn.means_ = res_cat[0][0]
gmm_morn.covars_ = res_cat[0][1]
gmm_morn.weights_ = res_cat[0][2][1]
Z_morn = gmm_morn.score_samples(XX)[0]
Z_morn = Z_morn.reshape(X.shape)

gmm_night = GMM(n_components=k, covariance_type='full', min_covar=1e-8)
gmm_night.means_ = res_cat[0][0]
gmm_night.covars_ = res_cat[0][1]
gmm_night.weights_ = res_cat[0][2][5]
Z_night = gmm_night.score_samples(XX)[0]
Z_night = Z_night.reshape(X.shape)

fig_morn = plt.figure(figsize=(6, 5.4))
ax = fig_morn.add_subplot(1, 1, 1)
ax.scatter(sd[:,0], sd[:,1], alpha=0.08, lw=0.1, s=10, color='0.8')
cs = ax.contour(X, Y, Z_morn)
ax.set_xlim(-74.05, -73.75)
ax.set_ylim(40.55, 40.9)
ax.set_xticks(np.arange(-74, -73.76, .05))
fig_morn.savefig('./include/gmm_cat_morn.png', bbox_inches='tight', dpi=150)

fig_night = plt.figure(figsize=(6, 5.4))
ax = fig_night.add_subplot(1, 1, 1)
ax.scatter(sd[:,0], sd[:,1], alpha=0.08, lw=0.1, s=10, color='0.8')
ax.contour(X, Y, Z_night, levels=cs.levels)
ax.set_xlim(-74.05, -73.75)
ax.set_ylim(40.55, 40.9)
ax.set_xticks(np.arange(-74, -73.76, .05))
fig_night.savefig('./include/gmm_cat_night.png', bbox_inches='tight', dpi=150)


x2 = np.linspace(-74.03, -73.91)
y2 = np.linspace(40.69, 40.83)
X2, Y2 = np.meshgrid(x2, y2)
XX2 = np.array([X2.ravel(), Y2.ravel()]).T

Z2_morn = gmm_morn.score_samples(XX2)[0]
Z2_morn = Z2_morn.reshape(X2.shape)

Z2_night = gmm_night.score_samples(XX2)[0]
Z2_night = Z2_night.reshape(X2.shape)

fig_morn2 = plt.figure(figsize=(6, 5.4))
ax = fig_morn2.add_subplot(1, 1, 1)
ax.scatter(sd[:,0], sd[:,1], alpha=0.08, lw=0.1, s=10, color='0.8')
cs2 = ax.contour(X2, Y2, Z2_morn)
ax.set_xlim(-74.03, -73.91)
ax.set_ylim(40.69, 40.83)
fig_morn2.savefig('./include/gmm_cat_morn2.png', bbox_inches='tight', dpi=150)

fig_night2 = plt.figure(figsize=(6, 5.4))
ax = fig_night2.add_subplot(1, 1, 1)
ax.scatter(sd[:,0], sd[:,1], alpha=0.08, lw=0.1, s=10, color='0.8')
ax.contour(X2, Y2, Z2_night, levels=cs2.levels)
ax.set_xlim(-74.03, -73.91)
ax.set_ylim(40.69, 40.83)
fig_night2.savefig('./include/gmm_cat_night2.png', bbox_inches='tight', dpi=150)


# Evaluate out-of-sample fit
sd2 = np.load('./data/data_test.npy')
hour2 = np.load('./data/data_test_hour.npy')

hour_cat2 = np.zeros(len(hour2), dtype=int)           # Early morning
hour_cat2[np.logical_and( 7 <= hour2, hour2 <  9)] = 1 # Morning rush
hour_cat2[np.logical_and( 9 <= hour2, hour2 < 16)] = 2 # Work day
hour_cat2[np.logical_and(16 <= hour2, hour2 < 18)] = 3 # Evening rush
hour_cat2[np.logical_and(18 <= hour2, hour2 < 22)] = 4 # Evening
hour_cat2[np.logical_or( 22 <= hour2, hour2 <  2)] = 5 # Night

liks = np.zeros((6, 3))
for h in range(6):
    np.random.seed(2046)
    gmm_part = GMM(n_components=k, covariance_type='full', min_covar=1e-8)
    start = time()
    gmm_part.fit(sd[hour_cat==h])
    print(h, time() - start)

    gmm_cattemp = GMM(n_components=k, covariance_type='full', min_covar=1e-8)
    gmm_cattemp.means_ = res_cat[0][0]
    gmm_cattemp.covars_ = res_cat[0][1]
    gmm_cattemp.weights_ = res_cat[0][2][h]

    liks[h,0] = gmm_mod.score(sd2[hour_cat2==h]).sum()
    liks[h,1] = gmm_part.score(sd2[hour_cat2==h]).sum()
    liks[h,2] = gmm_cattemp.score(sd2[hour_cat2==h]).sum()
