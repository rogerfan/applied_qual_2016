from datetime import datetime as dt

import numpy as np
import pandas as pd


data_yellow = pd.read_csv('./data/raw_data/yellow_tripdata_2015-12.csv')
data_green = pd.read_csv('./data/raw_data/green_tripdata_2015-12.csv')

drop_cols_yellow = [
    'VendorID', 'store_and_fwd_flag', 'extra', 'mta_tax',
    'tip_amount', 'tolls_amount', 'improvement_surcharge',
]
drop_cols_green = [
    'VendorID', 'Store_and_fwd_flag', 'Extra', 'MTA_tax',
    'Tip_amount', 'Tolls_amount', 'Ehail_fee', 'improvement_surcharge',
    'Trip_type ',
]
data_yellow = data_yellow.drop(drop_cols_yellow, axis=1)
data_green = data_green.drop(drop_cols_green, axis=1)

data_yellow.columns = [
    'pickup_time', 'dropoff_time', 'num_pass', 'trip_dist',
    'pickup_x', 'pickup_y', 'ratecode', 'dropoff_x', 'dropoff_y',
    'pay_type', 'fare_amt', 'tot_amt',
]
data_green.columns = [
    'pickup_time', 'dropoff_time', 'ratecode',
    'pickup_x', 'pickup_y', 'dropoff_x', 'dropoff_y',
    'num_pass', 'trip_dist', 'fare_amt', 'tot_amt', 'pay_type',
]

data = pd.concat([data_yellow, data_green])

data['pickup_time'] = pd.to_datetime(data['pickup_time'])
data['dropoff_time'] = pd.to_datetime(data['dropoff_time'])

data.to_pickle('./data/data_all.p')


sdata = pd.read_pickle('./data/data_all.p')

print(sdata.shape[0])
# 13068863
sdata = sdata[sdata['pickup_time'] >= '2015-12-07 05:00:00']
sdata = sdata[sdata['pickup_time'] < '2015-12-11 00:00:00']

print(sdata.shape[0])
print(np.sum(np.logical_or(sdata['dropoff_y'] == 0., sdata['pickup_y'] == 0.)))
# 1714343
#   25593
sdata = sdata[sdata['dropoff_y'] != 0.]
sdata = sdata[sdata['pickup_y'] != 0.]

print(sdata.shape[0])
# 1688750
sdata = sdata[sdata['pickup_x']  < -73]
sdata = sdata[sdata['dropoff_x'] < -73]
sdata = sdata[sdata['pickup_x']  > -75]
sdata = sdata[sdata['dropoff_x'] > -75]
sdata = sdata[sdata['pickup_y']  > 40.2]
sdata = sdata[sdata['dropoff_y'] > 40.2]
sdata = sdata[sdata['pickup_y']  < 42]
sdata = sdata[sdata['dropoff_y'] < 42]
print(sdata.shape[0])
# 1688673

sd = np.array(sdata[['pickup_x', 'pickup_y']])
hour = sdata['pickup_time'].dt.hour

sdata.to_pickle('./data/data_week.p')
np.save('./data/data_week_c.npy', sd)
np.save('./data/data_week_hour.npy', hour)


# Test data.
sdata2 = pd.read_pickle('./data/data_all.p')

sdata2 = sdata2[sdata2['pickup_time'] >= '2015-12-14 22:00:00']
sdata2 = sdata2[sdata2['pickup_time'] < '2015-12-15 02:00:00']

sdata2 = sdata2[sdata2['dropoff_y'] != 0.]
sdata2 = sdata2[sdata2['pickup_y'] != 0.]

sdata2 = sdata2[sdata2['pickup_x']  < -73]
sdata2 = sdata2[sdata2['dropoff_x'] < -73]
sdata2 = sdata2[sdata2['pickup_x']  > -75]
sdata2 = sdata2[sdata2['dropoff_x'] > -75]
sdata2 = sdata2[sdata2['pickup_y']  > 40.2]
sdata2 = sdata2[sdata2['dropoff_y'] > 40.2]
sdata2 = sdata2[sdata2['pickup_y']  < 42]
sdata2 = sdata2[sdata2['dropoff_y'] < 42]
print(sdata2.shape[0])
# 1688673

sd2 = np.array(sdata2[['pickup_x', 'pickup_y']])
np.save('./data/data_week_test.npy', sd2)


