from datetime import datetime as dt

import numpy as np
import pandas as pd


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

# data['pickup_time'] = pd.to_datetime(data['pickup_time'])
# data['dropoff_time'] = pd.to_datetime(data['dropoff_time'])

# data.to_pickle('./data/data_all.p')


sdata = pd.read_pickle('./data/data_all.p')

print(sdata.shape[0])
# 11460573
sdata = sdata[sdata['pickup_time'] >= '2015-12-07 05:00:00']
sdata = sdata[sdata['pickup_time'] < '2015-12-11 18:00:00']

print(sdata.shape[0])
print(np.sum(np.logical_or(sdata['dropoff_y'] == 0., sdata['pickup_y'] == 0.)))
# 1807479
#   29490
sdata = sdata[sdata['dropoff_y'] != 0.]
sdata = sdata[sdata['pickup_y'] != 0.]

print(sdata.shape[0])
# 1777989
sdata = sdata[sdata['pickup_x'] < -73]
sdata = sdata[sdata['dropoff_x'] < -73]
sdata = sdata[sdata['pickup_x'] > -74.3]
sdata = sdata[sdata['dropoff_x'] > -74.3]
sdata = sdata[sdata['pickup_y'] > 39.5]
sdata = sdata[sdata['dropoff_y'] > 39.5]
print(sdata.shape[0])
# 1777717

# np.random.seed(2634)
# inds = np.random.choice(sdata.shape[0], 100000, replace=False)
# sdata = data.iloc[inds]

sd = np.array(sdata[['pickup_x', 'pickup_y']])
hour = sdata['pickup_time'].dt.hour

sdata.to_pickle('./data/data_week.p')
np.save('./data/data_week_c.npy', sd)
np.save('./data/data_week_hour.npy', hour)
