
# ## Import Libraries & Data
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import datetime
import math

#dataset = pd.read_csv('data/2017-06-potong-1-new-freq.csv')
#dataset = pd.read_csv('data/2017-06-potong-2-new-freq.csv')
#dataset = pd.read_csv('data/2017-06-potong-2a-new-freq.csv')
dataset = pd.read_csv('data/2017-06-potong-3-new-freq.csv')

data = dataset.copy()


# ## Dealing with time
data['timestamp'] = pd.to_datetime(dataset['timestamp'])

def get_day_of_week(dt):
    return dt.weekday()
def get_hour(dt):
    return dt.hour

data['day_of_week'] = data['timestamp'].apply(get_day_of_week)
data['hour'] = data['timestamp'].apply(get_hour)


# ## Add location zone
data['location_zone'] = (data['linear_ref'] * 10000).apply(math.floor)
data[data['location_zone'] > 10000] = 10000
data[data['location_zone'] < 0] = 0


# ## Duplicate data for training/testing

# # 3nd previous point
temp_data3 = data.copy()
temp_data3['last_point_time'] = temp_data3['timestamp'].shift(3)
temp_data3['second_from_last_point'] = (temp_data3['timestamp'] - temp_data3['timestamp'].shift(3)).dt.total_seconds()
temp_data3['distance_from_last_point'] = temp_data3['linear_ref'] - temp_data3['linear_ref'].shift(3)
temp_data3['last_point_location'] = temp_data3['linear_ref'].shift(3)
temp_data3['last_point_lat'] = temp_data3['lat'].shift(3)
temp_data3['last_point_lon'] = temp_data3['lon'].shift(3)

# # 2nd previous point
temp_data = data.copy()
temp_data['last_point_time'] = temp_data['timestamp'].shift(2)
temp_data['second_from_last_point'] = (temp_data['timestamp'] - temp_data['timestamp'].shift(2)).dt.total_seconds()
temp_data['distance_from_last_point'] = temp_data['linear_ref'] - temp_data['linear_ref'].shift(2)
temp_data['last_point_location'] = temp_data['linear_ref'].shift(2)
temp_data['last_point_lat'] = temp_data['lat'].shift(2)
temp_data['last_point_lon'] = temp_data['lon'].shift(2)


# # Last point
data['last_point_time'] = data['timestamp'].shift()
data['second_from_last_point'] = (data['timestamp'] - data['timestamp'].shift()).dt.total_seconds()
data['distance_from_last_point'] = data['linear_ref'] - data['linear_ref'].shift()
data['last_point_location'] = data['linear_ref'].shift()
data['last_point_lat'] = data['lat'].shift()
data['last_point_lon'] = data['lon'].shift()


# ## append all the data
data = pd.concat([data, temp_data, temp_data3])


# ## Remove Missing data and Outlier
data.dropna(inplace=True)
data = data.drop(data[data['distance_from_last_point'] <= 0].index)
data = data.drop(data[data['distance_from_last_point'] > 0.2].index)
data = data.drop(data[data['second_from_last_point'] > 1000].index)


# ## Filter relavant data
data = data[['linear_ref', 'direction', 'day_of_week', 'hour', 'speed',
             'location_zone', 'second_from_last_point']]
#data = data[['lat', 'lon', 'direction', 'day_of_week', 'hour', 'speed',
#             'second_from_last_point',  'last_point_lat', 'last_point_lon']]

#data = data[['distance_from_last_point', 'direction', 'day_of_week', 'hour', 'speed',
#             'second_from_last_point', 'last_point_location']]


# ## Split Features and Label
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values


# ## Encode data & Train/Test split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


# ## Machine Learning Algorithm
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor()
# regressor.fit(X_train, y_train)

from sklearn.linear_model import Ridge
regressor = Ridge()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
score = regressor.score(X_test, y_test)


BUS_LINES = ['1', '2', '2a', '3']

def export_model(modellers, filename):
    from sklearn.externals import joblib
    joblib.dump(modellers, 'pickled-data2/'+filename+'.pkl')

def import_model(filename):
    from sklearn.externals import joblib
    return joblib.load('pickled-data/'+filename+'.pkl')

def save_model(filename):
    export_model([regressor, labelencoder, onehotencoder], filename)

def clean_data(data_point, time):
    new_data_point = data_point.copy()
    new_data_point['timestamp'] = pd.to_datetime(new_data_point['timestamp'])
    new_data_point['day_of_week'] = new_data_point['timestamp'].weekday()
    new_data_point['hour'] = new_data_point['timestamp'].hour
    new_data_point['second_from_last_point'] = (time - new_data_point['timestamp']).total_seconds()
#    new_data_point['last_point_location'] = new_data_point['linear_ref']
    new_data_point['last_point_lat'] = new_data_point['lat']
    new_data_point['last_point_lon'] = new_data_point['lon']
#    data_point.timestamp + datetime.timedelta(0,5)
    new_data_point['location_zone'] = math.floor(new_data_point['linear_ref'] * 10000)

    new_data_point = new_data_point[['direction', 'day_of_week', 'hour', 'speed', 
                                     'location_zone', 'second_from_last_point']]
    return new_data_point
    
def encode_data(data_point, labelencoder, onehotencoder):
    new_data_point = data_point.copy()
    new_data_point[0] = labelencoder.transform([new_data_point[0]])[0]
    new_data_point = onehotencoder.transform([new_data_point]).toarray()
    new_data_point = new_data_point[0, 1:]
    
    return new_data_point

def get_lastest_gps(bus_line, bus_vehicle_id):
    import requests
    data = requests.get('https://api.traffy.xyz/vehicle/?vehicle_id='+str(bus_vehicle_id)).json()
    bus = data['results']
    if(bus):
        return bus[0]
    return None


def get_bus_info(bus):
    bus_data = pd.Series()
    bus_data['vehicle_id'] = bus['vehicle_id']
    bus_data['timestamp'] = bus['info']['gps_timestamp']
#    bus_data['trip_id'] 
    bus_data['linear_ref'] = bus['checkin_data']['route_linear_ref']
    bus_data['speed'] = bus['info']['speed']
    bus_data['direction'] = bus['info']['direction']
    [bus_data['lat'], bus_data['lon']] = bus['info']['coords']
#    bus_data['route_length_in_meter']
#    bus_data['distance_from_route_in_meter']
    return bus_data

#encoded_bus_data = []

def predict_location(bus_line, bus_vehicle_id):
#    print(bus_line,bus_id)
    bus_line = str(bus_line)
    if(bus_line not in BUS_LINES):
        return 'This bus line is not available!'
        
#    print(pd.to_datetime(datetime.datetime.utcnow()))
    bus = get_lastest_gps(bus_line, bus_vehicle_id)
    
    if(not bus):
        return 'This bus is not available!'
        
#    [regressor, labelencoder, onehotencoder] = import_model('potong-' + bus_line)

    bus_data = get_bus_info(bus)
    time = pd.to_datetime(datetime.datetime.utcnow())
    cleaned_bus_data = clean_data(bus_data, time)
    encoded_bus_data = encode_data(cleaned_bus_data, labelencoder, onehotencoder)
    
#    np.set_printoptions(threshold=np.nan)
#    print(encoded_bus_data)
    
    predicted_location = regressor.predict([encoded_bus_data])
#    print(cleaned_bus_data['second_from_last_point'])
#    print(bus_data['linear_ref'])
    output = {'predicted_linear_ref': predicted_location[0],
              'predicting time': time,
                    'last_point_data': {
                        'last_timestamp': bus_data['timestamp'],
                        'timestamp_now': time,
                        'second_from_last_point': cleaned_bus_data['second_from_last_point'],
                        'last_linear_ref': bus_data['linear_ref'],
                        'last_speed': bus_data['speed'],
                        'direction': bus_data['direction'],}
                    }
    
#    output = jsonify(output)
  
    return output

#save_model('potong-3')