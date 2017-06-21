# # Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

dataset = pd.read_csv('data/2017-06-potong-1-new-freq.csv')
data = dataset.copy()

# # Dealing with time
data['timestamp'] = pd.to_datetime(dataset['timestamp'])

def get_day_of_week(dt):
    return dt.weekday()
def get_hour(dt):
    return dt.hour

data['day_of_week'] = data['timestamp'].apply(get_day_of_week)
data['hour'] = data['timestamp'].apply(get_hour)

#data['first_point'] = (data['trip_id'].shift() != data['trip_id'])

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

# # append all the data
data = pd.concat([data, temp_data, temp_data3])

# # linear ref delta
#data['delta_linear_ref'] = data['linear_ref'] - data[']

# # Remove Missing data and Outlier
data.dropna(inplace=True)
data = data.drop(data[data['distance_from_last_point'] <= 0].index)
data = data.drop(data[data['distance_from_last_point'] > 0.2].index)
data = data.drop(data[data['second_from_last_point'] > 1000].index)

# # Filter relavant data and divide into in and out trip
data = data[['linear_ref', 'direction', 'day_of_week', 'hour', 'speed',
             'second_from_last_point',  'last_point_location']]
#data = data[['lat', 'lon', 'direction', 'day_of_week', 'hour', 'speed',
#             'second_from_last_point',  'last_point_lat', 'last_point_lon']]

#data = data[['distance_from_last_point', 'direction', 'day_of_week', 'hour', 'speed',
#             'second_from_last_point', 'last_point_location']]

# # Machine Learning - Train data
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_direction = LabelEncoder()
X[:, 0] = labelencoder_direction.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
score = regressor.score(X_test, y_test)

def export_model(regressor, filename):
    from sklearn.externals import joblib
    joblib.dump(regressor, 'pickled-data/'+filename+'.pkl')

def import_model(filename):
    from sklearn.externals import joblib
    return joblib.load('pickled-data/'+filename+'.pkl')


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

    new_data_point = new_data_point[['direction', 'day_of_week', 'hour', 'speed', 
                                     'second_from_last_point', 'linear_ref']]
    return new_data_point
    
def encode_data(data_point):
    new_data_point = data_point.copy()
#    labelencoder_direction = LabelEncoder()
    new_data_point[0] = labelencoder_direction.transform([new_data_point[0]])[0]
    
#    onehotencoder = OneHotEncoder(categorical_features = [1])
    new_data_point = onehotencoder.transform([new_data_point]).toarray()
#    print(new_data_point.shape)
    new_data_point = new_data_point[0, 1:]
    
    return new_data_point
    
def get_lastest_gps(bus_id):
    import requests
    data = requests.get('https://api.traffy.xyz/vehicle/?line=potong-1').json()
    buses = data['results']
    for bus in buses:
        if(bus['id'] == int(bus_id)):
#            print('found the bus')
            return bus
#    print('Not found the Bus')
    return None

def get_bus_info(bus):
    bus_data = pd.Series()
    bus_data['vid'] = bus['id']
    bus_data['timestamp'] = bus['info']['gps_timestamp']
#    bus_data['trip_id'] 
    bus_data['linear_ref'] = bus['checkin_data']['route_linear_ref']
    bus_data['speed'] = bus['info']['speed']
    bus_data['direction'] = bus['info']['direction']
    [bus_data['lat'], bus_data['lon']] = bus['info']['coords']
#    bus_data['route_length_in_meter']
#    bus_data['distance_from_route_in_meter']
    return bus_data

def predict_location(bus_id):
    bus = get_lastest_gps(bus_id)
    if(not bus):
        return 'This bus is not available'
    bus_data = get_bus_info(bus)
    time = pd.to_datetime(datetime.datetime.utcnow())
    cleaned_bus_data = clean_data(bus_data, time)
    encoded_bus_data = encode_data(cleaned_bus_data)
    
    predicted_location = regressor.predict([encoded_bus_data])
    print(cleaned_bus_data['second_from_last_point'])
    print(bus_data['linear_ref'])
    return predicted_location

#def ttbb(bus_id):
#    bus = get_lastest_gps(bus_id)
#    if(not bus):
#        return 'This bus is not available'
#    bus_data = get_bus_info(bus)
#    time = pd.to_datetime(datetime.datetime.utcnow())
#    cleaned_bus_data = clean_data(bus_data, time)
#    encoded_bus_data = encode_data(cleaned_bus_data)
#    return cleaned_bus_data





