from os.path import isfile
from requests import get


def download(url, file):
    r = get(url, stream=True)
    with open(file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return file


to_dl = [
    ('yellow', '2015', '12'),
    ('green', '2015', '12'),
]

for col, yr, mn in to_dl:
    if isfile('./data/raw_data/{}_tripdata_{}-{}.csv'.format(col, yr, mn)):
        print('{}-{}-{} already downloaded.'.format(col, yr, mn))
    else:
        print('Downloading {}-{}-{}.'.format(col, yr, mn))
        url = 'https://storage.googleapis.com/tlc-trip-data/' \
              '{yr}/{col}_tripdata_{yr}-{mn}.csv'.format(col=col, yr=yr, mn=mn)
        file = './data/raw_data/{}_tripdata_{}-{}.csv'.format(col, yr, mn)
        download(url, file)

print('Finished downloading files.')
