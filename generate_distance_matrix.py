import pandas as pd
import h5py
import numpy as np
from collections import namedtuple
import csv
import requests


taxi_zone_lookup_chart = "/home/erynqian/10701/19F10701_Project/taxi _zone_lookup.csv"
datafields = ["VendorID","tpep_pickup_datetime","tpep_dropoff_datetime","passenger_count","PULocationID","DOLocationID","payment_type"]
# key = "6fW8tAG2L3VWCJYeKQ0IwgBzJNBJpoDZ"
Google_API_key = "AIzaSyCiEIaf3AW7Q6dyd90GHkWEUDfswzkCImg"
all_locations = []
distance_matrix = np.load("distance_matrix.npy")

def location_to_string(l):
    l[1] = '+'.join(l[1].split(' '))
    l[2] = '+'.join(l[2].split(' '))
    return l[2] + ',' + l[1] + ',NY'

def fetch_distance(loc1, loc2):
    url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins=" + \
          location_to_string(loc1) + "&destinations=" + location_to_string(loc2) +"&key=" + Google_API_key
    response = requests.get(url)
    result = float(response.json()['rows'][0]['elements'][0]['distance']['text'].split(' ')[0])
    print(result)
    return result

"""Load taxi zone lookup chart"""
with open(taxi_zone_lookup_chart) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    first = 0
    for row in csv_reader:
        if first == 0:
            first += 1
        else:
            all_locations.append(row)
all_locations = all_locations[:-2]

# print(distance_matrix.shape)
# for i in range(228,263):
#     for j in range(i+1, 263):
#         print(i,j, end=" = ")
#         distance_matrix[i][j] = fetch_distance(all_locations[i], all_locations[j])
#         if j % 50 == 0:
#             np.save("distance_matrix", distance_matrix)        
#     np.save("distance_matrix", distance_matrix)
 

# TEST
y = np.load("distance_matrix.npy")
print(y[:10])