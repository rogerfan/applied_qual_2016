from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# data.to_pickle('./data/data.p')
data = pd.read_pickle('./data/data.p')


np.random.seed(2634)
inds = np.random.choice(data.shape[0], 1000000, replace=False)
sdata = data.iloc[inds]

sdata = sdata[sdata['pickup_x'] < -72.]
sdata = sdata[sdata['pickup_x'] > -74.3]
sdata = sdata[sdata['dropoff_x'] < -72.]
sdata = sdata[sdata['dropoff_x'] > -74.3]

sd = np.array(sdata[['pickup_x', 'pickup_y']])



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
        logliks.append(np.sum(probs * logpdfs))

        # M-step
        for k in range(len(init_mu)):
            curr_mu[k] = np.average(data, axis=0, weights=probs[:,k])
            # curr_sigma[k] = np.cov(data, rowvar=0, aweights=probs[:,k], ddof=0)

            dm = data - curr_mu[k]
            curr_sigma[k] = np.dot(probs[:,k]*dm.T, dm) / np.sum(probs[:,k])
        curr_pi = np.mean(probs, axis=0)

        if num_iter >= 2:
            diff = np.abs(logliks[-1] - logliks[-2])
        num_iter += 1

    logpdfs = calc_logpdfs(data, curr_mu, curr_sigma)
    probs = calc_probs(logpdfs, curr_pi)
    logliks.append(np.sum(probs * logpdfs))

    return (curr_mu, curr_sigma, curr_pi), probs, logliks

def init_pp(data, num_clusters):
    n = len(data)
    init = np.zeros((num_clusters, data.shape[1]))
    init[0] = data[np.random.randint(n)]

    for i in range(1, num_clusters):
        distances = np.zeros((n, i))
        for j in range(i):
            distances[:,j] = la.norm(data - init[j, None], axis=1)

        min_ind = np.argmin(distances, axis=1)
        min_dist = distances[np.arange(n), min_ind]

        init[i] = data[np.random.choice(n, p=min_dist/np.sum(min_dist))]

    return(init)



from matplotlib import cm
colors = cm.Paired(np.linspace(0, 1, 10))

km = KMeans(8)
km.fit(sd)

pred = km.predict(sd)
fig = plt.figure()
ax = fig.add_subplot(111)
for g, c in zip(range(10), colors):
    ax.scatter(sd[pred==g, 0], sd[pred==g, 1], c=c)




# Run GMM
init_sigma = [np.identity(2) for i in range(8)]
init_pi = np.array([1 for i in range(8)])/8

np.random.seed(2046)
best_gmm = (None, None, [-np.inf])
run, iters, logliks = [], [], []
start = time()
for i in range(1):
    init = km.cluster_centers_
    res = gmm(sd, init, init_sigma, init_pi, max_iter=2000)

    run += [i]*len(res[2])
    iters += list(range(len(res[2])))
    logliks += res[2]

    if res[2][-1] > best_gmm[2][-1]:
        best_gmm = res
print(time() - start)



fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(sd[:10000, 0], sd[:10000, 1])
ax.scatter(best_gmm[0][0][:,0], best_gmm[0][0][:,1], color='red')

np.savetxt('./data/scoords.txt', sd)


