from time import time
from datetime import datetime as dt

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


# data = pd.read_csv('./data/raw_data/yellow_tripdata_2015-12.csv')

# drop_cols = [
#     'VendorID', 'store_and_fwd_flag', 'extra', 'mta_tax',
#     'tip_amount', 'tolls_amount', 'improvement_surcharge',
# ]

# data = data.drop(drop_cols, axis=1)
# data.columns = [
#     'pickup_time', 'dropoff_time', 'num_pass', 'trip_dist',
#     'pickup_x', 'pickup_y', 'ratecode', 'dropoff_x', 'dropoff_y',
#     'pay_type', 'fare_amt', 'tot_amt',
# ]

# print(data.shape[0])
# print(np.sum(np.logical_or(data['dropoff_y'] == 0., data['pickup_y'] == 0.)))
# # 11460573
# #   184633
# data = data[data['dropoff_y'] != 0.]
# data = data[data['pickup_y'] != 0.]

# data['pickup_time'] = pd.to_datetime(data['pickup_time'])
# data['dropoff_time'] = pd.to_datetime(data['dropoff_time'])

# data.to_pickle('./data/data.p')
data = pd.read_pickle('./data/data.p')



sdata = data.copy()
sdata = sdata[sdata['pickup_time'] >= '2015-12-07 05:00:00']
sdata = sdata[sdata['pickup_time'] < '2015-12-11 06:00:00']

sdata = sdata[sdata['pickup_x'] < -73]
sdata = sdata[sdata['dropoff_x'] < -73]
sdata = sdata[sdata['pickup_x'] > -74.3]
sdata = sdata[sdata['dropoff_x'] > -74.3]
sdata = sdata[sdata['pickup_y'] > 39.5]
sdata = sdata[sdata['dropoff_y'] > 39.5]

np.random.seed(2634)
inds = np.random.choice(sdata.shape[0], 100000, replace=False)
sdata = data.iloc[inds]

sd = np.array(sdata[['pickup_x', 'pickup_y']])



k = 10

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


# Run GMM
init_sigma = [np.cov(sd, rowvar=0) for i in range(k)]
init_pi = np.array([1 for i in range(k)])/k
init_mu = km.cluster_centers_

np.random.seed(2046)
start = time()
res = gmm(sd, init_mu, init_sigma, init_pi, diff_tol=1e-2, max_iter=1000)
print(time() - start)

iters = list(range(len(res[2])))
logliks = res[2]




fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(sd[:,0], sd[:,1])
ax.scatter(res[0][0][:,0], res[0][0][:,1], color='red')

np.savetxt('./data/scoords.txt', sd)


