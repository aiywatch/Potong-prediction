#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:12:06 2017

@author: aiy
"""
from sklearn.externals import joblib
import pandas as pd
import datetime

BUS_LINES = ['1', '2', '2a', '3']

def export_model(regressor, filename):
    joblib.dump(regressor, 'pickled-data/'+filename+'.pkl')

def import_model(filename):
    return joblib.load('pickled-data/'+filename+'.pkl')

#[regressor, labelencoder_direction, onehotencoder] = import_model('potong-1')


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
    
def encode_data(data_point, labelencoder, onehotencoder):
    new_data_point = data_point.copy()
    new_data_point[0] = labelencoder.transform([new_data_point[0]])[0]
    new_data_point = onehotencoder.transform([new_data_point]).toarray()
    new_data_point = new_data_point[0, 1:]
    
    return new_data_point
    
def get_lastest_gps(bus_line, bus_id):
    import requests
    data = requests.get('https://api.traffy.xyz/vehicle/?line=potong-'+str(bus_line)).json()
    buses = data['results']
    for bus in buses:
        if(bus['id'] == int(bus_id)):
            return bus
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

def predict_location(bus_line, bus_id):
    bus_line = str(bus_line)
    if(bus_line not in BUS_LINES):
        return 'This bus line is not available!'
        
    bus = get_lastest_gps(bus_line, bus_id)
    
    if(not bus):
        return 'This bus is not available!'
        
    [regressor, labelencoder, onehotencoder] = import_model('potong-' + bus_line)

    bus_data = get_bus_info(bus)
    time = pd.to_datetime(datetime.datetime.utcnow())
    cleaned_bus_data = clean_data(bus_data, time)
    encoded_bus_data = encode_data(cleaned_bus_data, labelencoder, onehotencoder)
    
    
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

