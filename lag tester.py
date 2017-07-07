import pandas as pd
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib


data = pd.read_csv('check_potong/GPS_2017.07.03 10:31:13.640-059049461.csv')
raw_time_data = pd.read_csv('check_potong/dtc_with_speed.csv')


# include raw time

dtc_data = raw_time_data[(raw_time_data['r_time'] > '2017-07-03 10:00') & (raw_time_data['r_time'] < '2017-07-03 11:30')]

raw_time = []
found = []
speed = []
for d in data.iterrows():
    matched = dtc_data[dtc_data['r_lat'] == d[1]['lat_FromDTC']]
    raw_time.append(matched.iloc[-1, :]['r_time'])
    speed.append(matched.iloc[-1, :]['r_speed'])
    found.append(matched.shape[0])

data['dtc_raw_time'] = raw_time
data['dtc_speed'] = speed
data['dtc_raw_time'] = pd.to_datetime(data['dtc_raw_time'])






# time

data['log_timestamp'] = pd.to_datetime(data['log_timestamp'])
data['GNSS_TimeStamp'] = pd.to_datetime(data['GNSS_TimeStamp'])
data['DTC_TimeStamp'] = pd.to_datetime(data['DTC_TimeStamp'])

seven_hour = datetime.timedelta(hours=7)

gnss_diff = data['log_timestamp'] - data['GNSS_TimeStamp'] - seven_hour
dtc_diff = data['log_timestamp'] - data['dtc_raw_time']


gnss_diff.iloc[21:].mean()
dtc_diff[3:].mean()


dtc_diff = data['log_timestamp'] - data['dtc_raw_time']

plt.scatter(range(len(dtc_diff)-6), dtc_diff[6:].dt.seconds, c='blue', alpha=0.5, label="DTC")
#plt.scatter(range(len(gnss_diff)-3), gnss_diff[3:].dt.seconds, c='red', alpha=0.5, label="GNSS")
plt.xlabel("Nth point of collected data")
plt.ylabel("lag time (Seconds)")
plt.title("Lag time")
plt.legend()
plt.show()



# convert to linref from API
#linref = []
#
#import time
#
#for i in range(data.shape[0]):
##for i in range(100):
#    lat, lng = data.loc[i, ['lat_FromGnss', 'lng_FromGnss']]
#    req = requests.get('https://api.traffy.xyz/v0/route/583/linear_ref/?coords={},{}'.format(lng, lat)).json()
#    lin_gnss = req['location']['linear_ref']
#    
#    lat, lng = data.loc[i, ['lat_FromDTC', 'lng_FromDTC']]
#    req = requests.get('https://api.traffy.xyz/v0/route/583/linear_ref/?coords={},{}'.format(lng, lat)).json()
#    lin_dtc = req['location']['linear_ref']
#    
#    lat, lng = data.loc[i, ['lat_FromDevice', 'lng_FromDevice']]
#    req = requests.get('https://api.traffy.xyz/v0/route/583/linear_ref/?coords={},{}'.format(lng, lat)).json()
#    lin_device = req['location']['linear_ref']
#
#    linref.append([lin_gnss, lin_dtc, lin_device])
#    print('round ' + str(i) )
#    time.sleep(1)
#
#
#with open('pickled-lag/linref','wb') as f:
#    pickle.dump(linref,f)

linref = pickle.load(open('pickled-lag/linref', 'rb'))


lin_collected = pd.DataFrame(linref, columns=('gnss', 'dtc', 'device'))

diff_gnss = lin_collected['gnss'] - lin_collected['device']
diff_dtc = lin_collected['dtc'] - lin_collected['device']

diff_gnss_meter = diff_gnss * 13883
diff_dtc_meter = diff_dtc * 13883

diff_gnss_meter.mean()
diff_dtc_meter.mean()


#import matplotlib.pyplot as plt
#plt.plot(range(100), diff_gnss_meter.abs(), color='red')
#plt.plot(range(100), diff_dtc_meter.abs(), color='blue')
#
#plt.show()
#
#
#plt.plot(range(len(gnss_diff)), gnss_diff.dt.seconds, color='red')
#plt.plot(range(len(dtc_diff)), dtc_diff.dt.seconds, color='blue')
#
#plt.show()
#
#plt.scatter(range(len(gnss_diff)-3), gnss_diff[3:].dt.seconds, color='red')
#plt.show()




def import_model(filename):
    return joblib.load('pickled-data/'+filename+'.pkl')
def get_day_of_week(dt):
    return dt.weekday()
def get_hour(dt):
    return dt.hour



[regressor, labelencoder, onehotencoder] = import_model('potong-2')


dtc_time_utc = data['dtc_raw_time'] - seven_hour
linref = np.array(linref)


dtc_test_set = pd.DataFrame()
dtc_test_set['direction'] = ['out'] * 598
dtc_test_set['day_of_week'] = dtc_time_utc.apply(get_day_of_week)
dtc_test_set['hour'] = dtc_time_utc.apply(get_hour)
dtc_test_set['speed'] = speed
dtc_test_set['linear_ref'] = linref[:, 1]
dtc_test_set['second_from_last_point'] = dtc_diff.apply(lambda d: d.seconds)

dtc_test_set.loc[:, 'direction'] = labelencoder.transform(dtc_test_set.loc[:, 'direction'])
dtc_test_set = onehotencoder.transform(dtc_test_set).toarray()
dtc_test_set = dtc_test_set[:, 1:]

dtc_predict = []
for dtc_point in dtc_test_set:
    dtc_predict.append(regressor.predict([dtc_point]))

dtc_predict = np.array(dtc_predict)

diff_predicted_dtc = dtc_predict[:, 0] - lin_collected['device']
diff_predicted_dtc_meter = diff_predicted_dtc * 13883

plt.plot(range(598-6), np.absolute(diff_predicted_dtc_meter[6:]), c='blue', label="DTC")
#plt.plot(range(598), diff_gnss_meter.abs(), c='red', label="GNSS")
plt.xlabel("Nth point of collected data")
plt.ylabel("lag Distance (meters)")
plt.title("Lag Distance from Actual")
plt.legend()
plt.grid()
plt.show()
